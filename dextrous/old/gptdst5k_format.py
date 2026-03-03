
import dataclasses
import dextrous.old as ez

import typing as T

slotlike: T.TypeAlias = T.Union[
    str,                        # name
    tuple[str, str],            # name, description
    tuple[str, list[str]],      # name, values
    tuple[str, str, list[str]], # name, description, values
    'Slot'
]
@dataclasses.dataclass(eq=False)
class Slot:
    categorical:bool=None
    def __init__(self,
        name: slotlike,
        description: str | None = None,
        domain: str|None = None,
        values: list[str]|None = None,
        categorical:bool|None = None
    ):
        if isinstance(name, Slot):
            self.name = name.name # noqa
            self.description = name.description # noqa
            self.domain = name.domain # noqa
            self.values = name.values # noqa
            self.categorical = name.categorical # noqa
            return
        if isinstance(name, tuple) and name:
            for field in name[1:]:
                if isinstance(field, str):
                    description = field
                elif isinstance(field, list):
                    values = field
            name = name[0]
        self.name:str = name
        self.description:str|None = description
        self.domain:str|None = domain
        self.values:list[str]|None = values
        self.categorical:bool = categorical
        if self.categorical is None and self.values:
            self.categorical = not (set(self.values) & {'Etc', 'etc', 'etc.', 'Etc.', '...'})

    def key(self):
        return self.name, self.description, self.domain

    def __eq__(self, other):
        return self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

    def __str__(self):
        return f'{type(self).__name__}({self.name})'
    __repr__ = __str__

ontologylike: T.TypeAlias = T.Union[
    list[slotlike],
    dict[str,str|list[str]|tuple[str,list[str]]],
    'Ontology'
]
class Ontology:

    def __init__(
        self,
        ontology:ontologylike=None,
        **domains:ontologylike
    ):
        self.slot_ids: dict[tuple[str, str | None, str | None], Slot] = getattr(
            self, 'slot_ids', {}
        )
        if ontology is not None:
            domains[None] = ontology  # noqa
        for domain, slots in domains.items():
            if isinstance(slots, dict):
                slotlist = []
                for name, definition in slots.items():
                    if isinstance(definition, str):
                        slotlist.append((name, definition, None))
                    elif isinstance(definition, list):
                        slotlist.append((name, None, definition))
                    elif isinstance(definition, tuple):
                        slotlist.append((name, *definition))
                for slot in slotlist:
                    self.add(slot, domain=domain)
            elif isinstance(slots, Ontology):
                self.slot_ids = slots.slot_ids
                for slot in ontology.slots():
                    self.add(slot)
    update = __init__

    def add(self, slot: slotlike, domain=None):
        slot = Slot(slot, domain=domain if domain is not None else getattr(slot, 'domain', None))
        slot_id = (slot.name, slot.description, slot.domain)
        if slot_id in self.slot_ids:
            vars(self.slot_ids[slot_id]).update({k:v for k, v in vars(slot).items() if v is not None})
        else:
            self.slot_ids[slot_id] = slot
        return self.slot_ids[slot_id]

    def slots(self) -> list[Slot]:
        return list(self.slot_ids.values())

    def domains(self) -> dict[str, list[Slot]]:
        domains = {}
        for slot in self.slots():
            domains.setdefault(slot.domain, []).append(slot)
        return domains

    def __str__(self):
        return f'<{type(self).__name__} object with {len(self.slot_ids)} slots>'
    __repr__ = __str__


turnlike: T.TypeAlias = T.Union[
    str,                                        # text
    tuple[str, dict[slotlike, str|list[str]]],  # text, slots
    tuple[str, dict[slotlike, str|list[str]], dict[slotlike, str|list[str]]], # text, slots, predicted slots
    'Turn'
]
class Turn:
    def __init__(self,
        turn:turnlike,
        speaker=None,
        listener=None,
        slots:dict[slotlike, str|list[str]]=None,
        predicted_slots:dict[slotlike, str|list[str]]=None,
        dialogue=None,
        index=None
    ):
        self.turn:str = None # noqa
        self.speaker:str|None = speaker
        self.listener:str|None = listener
        self.slots:dict[Slot, list[str]]|None = None
        self.predicted_slots:dict[Slot, list[str]]|None = predicted_slots
        if slots is not None:
            self.slots = {}
            for slot, value in slots.items():
                if not isinstance(slot, Slot):
                    slot = Slot(slot)
                if isinstance(value, str):
                    value = [value]
                self.slots[slot] = value
        self.dialogue:T.Optional['Dialogue'] = dialogue
        self.index:int|None = index
        if isinstance(turn, Turn):
            vars(self).update({
                k:v for k, v in vars(turn).items()
                if v is not None and getattr(self, k, None) is None
            })
        elif isinstance(turn, str):
            self.turn = turn
        elif isinstance(turn, tuple):
            self.turn = turn[0]
            self.slots:dict[Slot, list[str]]|None = {}
            for slot, value in turn[1].items():
                slot = Slot(slot)
                if isinstance(value, str):
                    value = [value]
                self.slots[slot] = value
            if len(turn) >= 3:
                self.predicted_slots = {}
                for slot, value in turn[2].items():
                    slot = Slot(slot)
                    if isinstance(value, str):
                        value = [value]
                    self.predicted_slots[slot] = value

    def context(self):
        return self.dialogue.turns[:self.index+1]

    def dialogue_state(self):
        state = {}
        for turn in self.context():
            if turn.slots:
                state.update(
                    (slot, values) for slot, values in turn.slots.items()
                    if slot not in state or values is not None
                )
        return state

    def predicted_dialogue_state(self):
        state = {}
        for turn in self.context():
            if turn.predicted_slots:
                state.update(
                    (slot, values) for slot, values in turn.predicted_slots.items()
                    if slot not in state or values
                )
        return state

    def display(
        self,
        window=1,
        entire_state=False,
        include_description=False,
        include_empty_slots=True,
        examples_limit=0
    ):
        context = self.context()[max(0, self.index-window+1):self.index+1]
        display = '\n'.join(turn.turn for turn in context)
        if self.slots:
            lines = []
            if entire_state:
                slots = self.dialogue_state().items()
            else:
                slots = self.slots.items() if self.slots else []
            for slot, values in slots:
                values = values or []
                if not include_empty_slots and not values:
                    continue
                description = slot.description or []
                description = description if include_description else ''
                if description:
                    description = '\n    ' + description # noqa
                exs = ', '.join(
                    slot.values if examples_limit is None else slot.values[:examples_limit]
                ) if slot.values else ''
                if exs:
                    exs = '\n    (' + exs + ')'
                lines.append(
                    f"  {slot.name}: {', '.join(values)}{description}{exs}"
                )
            display += '\n\n' + '\n'.join(lines)
        if self.predicted_slots:
            display += '\n\n' + '\n'.join(
                f"  {slot.name}: {', '.join(values)}"
                for slot, values in (
                    self.predicted_dialogue_state().items()
                    if entire_state else self.predicted_slots.items()
                ) if values is not None
            )
        return display

    def __str__(self):
        return f'{type(self).__name__}({self.turn}{" | " + str(len(self.slots)) + " slots" if self.slots else ""})'
    __repr__ = __str__


dialoguelike: T.TypeAlias = T.Union[
    turnlike,
    list[turnlike],
    'Dialogue'
]
class Dialogue:
    Turn = Turn
    def __init__(self, turns:dialoguelike=None):
        if turns is None:
            turns = []
        self.turns:list[Turn] = getattr(self, 'turns', [])
        self.data: T.Optional['DstData'] = getattr(self, 'data', None)
        if isinstance(turns, Dialogue):
            for turn in turns.turns:
                self.add(turn)
        elif isinstance(turns, list):
            for turn in turns:
                self.add(turn)
        else:
            self.add(turns)
    update = __init__

    def add(self, turnlike):
        self.turns.append(type(self).Turn(turnlike, dialogue=self, index=len(self.turns)))

    def domains(self):
        domains = set()
        for turn in self.turns:
            if turn.slots:
                for slot in turn.slots:
                    domains.add(slot.domain)
        return domains

    def __str__(self):
        unique_slots = set()
        for turn in self.turns:
            if turn.slots:
                unique_slots.update(slot.name for slot in turn.slots)
        return f'{type(self).__name__}({len(self.turns)} turns{", "+str(len(unique_slots))+" slots" if unique_slots else ""})'
    __repr__ = __str__


dstlike: T.TypeAlias = T.Union[
    turnlike,
    list[turnlike|Dialogue],
    list[list[turnlike]],
    'DstData'
]
class DstData(ez.Data):
    Dialogue = Dialogue
    def __init__(self, dialogues:dstlike, ontology:ontologylike=None, file:ez.filelike=None):
        self.dialogues:list[Dialogue] = getattr(self, 'dialogues', [])
        self.ontology:Ontology = (
            Ontology(ontology) if ontology else getattr(self, 'ontology', Ontology())
        )
        if isinstance(dialogues, DstData):
            file = dialogues.file if file is None else file
            for dialogue in dialogues.dialogues:
                self.add(dialogue)
        elif isinstance(dialogues, list):
            for dialogue in dialogues:
                self.add(dialogue)
        else:
            self.add(dialogues)
        self._file = file
        self.__post_init__()
    update = __init__

    def domains(self) -> dict[str, set[Dialogue]]:
        domains = {}
        for dialogue in self.dialogues:
            for turn in dialogue.turns:
                if turn.slots:
                    for slot in turn.slots:
                        domains.setdefault(slot.domain, set()).add(dialogue)
        return domains

    def add(self, dialogue:dialoguelike):
        dialogue = type(self).Dialogue(dialogue)
        dialogue.data = self
        self.dialogues.append(dialogue)
        for turn in dialogue.turns:
            if turn.slots:
                for slot, values in list(turn.slots.items()):
                    del turn.slots[slot]
                    slot = self.ontology.add(slot)
                    turn.slots[slot] = values
            if turn.predicted_slots:
                for slot, values in list(turn.predicted_slots.items()):
                    del turn.predicted_slots[slot]
                    slot = self.ontology.add(slot)
                    turn.predicted_slots[slot] = values
        return dialogue

    def __str__(self):
        return f'{type(self).__name__}({len(self.dialogues)} dialogues{" from "+str(self.file) if self.file else ""})'
    __repr__ = __str__