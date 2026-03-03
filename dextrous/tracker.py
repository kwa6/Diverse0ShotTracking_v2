
import ezpyzy as ez
import dataclasses as dc
from dataclasses import dataclass as settings; vars().update(settings=ez.settings)
from language_model.llama import Llama, LlamaHyperparameters
from language_model.t5 import T5, T5Hyperparameters

import dextrous.dst_data as dst
import dextrous.preprocessing as dpp

import itertools as it
import pathlib as pl
import difflib as dl
import random as rng

from tqdm import tqdm


@settings
class TrackerHyperparameters(ez.Settings):
    model_path: ez.ColStr = None
    neg_examples_ratio: ez.ColFloat = ez.Def(0.0)
    """this has a bug, downsampling eliminates positives which affects turn target candidates of negative samples; also, neg sampling doesn't respect the eval domains hyperparam"""
    empty_slot_token: ez.ColStr = ez.Def("n/a")
    is_categorical_limit: ez.ColInt = ez.Def(10)
    train_all_slots_per_domain: ez.ColBool = None
    train_all_slots_all_domains: ez.ColBool = None
    train_percent_with_description: ez.ColFloat = ez.Def(1.0)
    train_percent_description_only: ez.ColFloat = ez.Def(0.0)
    train_percent_with_value_exs: ez.ColFloat = ez.Def(0.0)
    train_percent_value_ex_includes_actual_value: ez.ColFloat = ez.Def(0.1)
    train_percent_with_categories: ez.ColFloat = ez.Def(0.0)
    train_filters_out_descriptions_with_actual_value: ez.ColBool = ez.Def(False)
    train_max_value_string_length: ez.ColStr = ez.Def(50)
    train_num_candidates_distribution: ez.Column[list[float]] | list[float] | None = ez.Def(lambda: [0.2]*5)
    eval_with_categories: ez.ColBool = ez.Def(True)
    eval_with_value_exs: ez.ColBool = ez.Def(True)
    request_token: ez.ColStr = ez.Def('?')
    max_candidate_value_string_length: ez.ColInt = ez.Def(30)
    slot_instruction_prefix: ez.ColStr = ez.Def("Identify the information from the above dialogue:")
    train_remove_request_token_percent: ez.ColFloat = ez.Def(0.0)
    prediction_lowercase: ez.ColBool = ez.Def(False)
    prediction_fuzzy_match_candidates: ez.ColBool = ez.Def(False)
    uncased: ez.ColBool = ez.Def(False)
    train_prop_add_continuation: ez.ColFloat = ez.Def(None)
    train_prop_add_continuation_pcent_existing: ez.ColFloat = ez.Def(None)
    train_continuation_exs_replace_original: ez.ColBool = ez.Def(False)
    exclude_speakers: ez.Column[list] | list[str] | None = ez.Def(None)



@settings
class LlamaTrackerHyperparameters(LlamaHyperparameters, TrackerHyperparameters, ez.Settings): pass

@settings
class T5TrackerHyperparameters(T5Hyperparameters, TrackerHyperparameters, ez.Settings): pass


class LlamaTracker:
    def __init__(self, hyperparameters: LlamaTrackerHyperparameters|T5TrackerHyperparameters):
        if isinstance(hyperparameters, LlamaHyperparameters):
            ModelHyperparameters = LlamaHyperparameters
            Model = Llama
        else:
            ModelHyperparameters = T5Hyperparameters
            Model = T5
        if hyperparameters.model_path is not None:
            path = ez.File(hyperparameters.model_path).path
            tracker_hyperparams = ez.File(path/'tracker_hyperparameters.json').load()
            assert isinstance(tracker_hyperparams, dict), \
                "Did you forget to specify the iteration of a loaded model?" \
                " Your model_path is probably incorrect."
            tracker_hyperparams.update(hyperparameters.settings)
            vars(hyperparameters).update(tracker_hyperparams)
            llama_args = {
                k: v for k, v in vars(hyperparameters).items()
                if k in {f.name for f in dc.fields(ModelHyperparameters)}
                and k in hyperparameters.settings
            }
            llama_args.update(base=str(path))
            self.model = Model(**llama_args)
        else:
            llama_args = {
                k: v for k, v in vars(hyperparameters).items()
                if k in {f.name for f in dc.fields(ModelHyperparameters)}
            }
            self.model = Model(**llama_args)
        self.hyperparameters = hyperparameters

    def save(self, path: ez.filelike):
        tracker_hyperparams_path = ez.File(path).path/'tracker_hyperparameters.json'
        hyperparameters = {k:v for k,v in vars(self.hyperparameters).items() if k != 'settings'}
        ez.File(tracker_hyperparams_path).save(hyperparameters)
        self.model.save(path)
        return path

    def preprocess(self, data: dst.Data, for_training=False):
        did_to_max_idx = {}
        for did, idx in zip(data.turns.dialogue, data.turns.turn_index):
            did_to_max_idx[did] = max(did_to_max_idx.get(did, -1), idx)
        did_idx_to_forbidden_neg_slots = {
            (did, idx): set() for did, idx in zip(data.turns.dialogue, data.turns.turn_index)}
        turn_id_to_did_idx = {turn_id: (did, idx) for turn_id, did, idx in zip(
            data.turns.turn_id, data.turns.dialogue, data.turns.turn_index
        )}
        turn_to_domain = {turn_id: domain for turn_id, domain in zip(data.turns.turn_id, data.turns.domain)}
        domain_to_tid = {}
        domain_to_sid = {}
        for tid, slot in zip(data.slot_values.turn_id, data.slot_values.slot):
            domain = turn_to_domain[tid]
            if domain not in domain_to_tid:
                domain_to_tid[domain] = []
            domain_to_tid[domain].append(tid)
            if domain not in domain_to_sid:
                domain_to_sid[domain] = set()
            domain_to_sid[domain].add(slot)
        _ = set()
        for turn_id, slot in zip(data.slot_values.turn_id, data.slot_values.slot):
            did, idx = turn_id_to_did_idx[turn_id]
            for history_idx in range(idx, did_to_max_idx[did]+1):
                did_idx_to_forbidden_neg_slots.get((did, history_idx), _).add(slot)
        if for_training and (
            self.hyperparameters.train_prop_add_continuation is not None or
            self.hyperparameters.train_prop_add_continuation_pcent_existing is not None or
            self.hyperparameters.train_continuation_exs_replace_original
        ):
            if self.hyperparameters.train_continuation_exs_replace_original:
                dpp.replace_to_continuation(data, self.hyperparameters.request_token)
            else:
                dpp.add_continuation_values(data,
                    pcent_of_continuation=self.hyperparameters.train_prop_add_continuation,
                    pcent_of_existing=self.hyperparameters.train_prop_add_continuation_pcent_existing,
                    req_token=self.hyperparameters.request_token)
        if self.hyperparameters.uncased:
            data.slot_values.value[:] = [
                x.lower() if isinstance(x, str) else x for x in data.slot_values.value
            ]
            data.slot_values.slot[:] = [
                x.lower() if isinstance(x, str) else x for x in data.slot_values.slot
            ]
            data.slots.slot[:] = [
                x.lower() if isinstance(x, str) else x for x in data.slots.slot
            ]
            data.slots.description[:] = [
                x.lower() if isinstance(x, str) else x for x in data.slots.description
            ]
            data.value_candidates.candidate_value[:] = [
                x.lower() if isinstance(x, str) else x for x in data.value_candidates.candidate_value
            ]
            data.turns.text[:] = [
                x.lower() if isinstance(x, str) else x for x in data.turns.text
            ]
        if for_training:
            if self.hyperparameters.train_remove_request_token_percent:
                to_delete = [
                    svid for svid, value in zip(data.slot_values.slot_value_id, data.slot_values.value)
                    if value.strip() == self.hyperparameters.request_token
                ]
                num_request_to_remove = int(len(to_delete) * self.hyperparameters.train_remove_request_token_percent)
                to_delete = rng.sample(to_delete, num_request_to_remove)
                del data.slot_values[to_delete]
                del data.predictions[to_delete]
            if self.hyperparameters.train_max_value_string_length:
                to_filter = []
                for svid, value in zip(data.slot_values.slot_value_id, data.slot_values.value):
                    if isinstance(value, str) and len(value) > self.hyperparameters.train_max_value_string_length:
                        to_filter.append(svid)
                del data.slot_values[to_filter]
                del data.predictions[to_filter]
            if self.hyperparameters.train_all_slots_all_domains:
                with ez.Timer('Adding negative slots (cross-domain)'):
                    dpp.add_neg_slot_targets(
                        data, per_domain=False, exclude_speakers=self.hyperparameters.exclude_speakers
                    )
            elif self.hyperparameters.train_all_slots_per_domain:
                with ez.Timer('Adding negative slots (within-domain)'):
                    dpp.add_neg_slot_targets(
                        data, per_domain=True, exclude_speakers=self.hyperparameters.exclude_speakers
                    )
        turns, values, slots, candidates, predictions = (
            data.turns, data.slot_values, data.slots, data.value_candidates, data.predictions
        )
        with ez.Timer('Grouping slots by domain'):
            domain_to_slots = {domain: set() for domain in slots.domain}
            for domain, slot_id in zip(slots.domain, slots.slot_id):
                domain_to_slots[domain].add(slot_id)
        with ez.Timer('Sorting turns by index'):
            turns = data.turns().sort(turns.turn_index)
        with ez.Timer('Grouping turns by dialogue'):
            dial_to_turns = {dialogue: [] for dialogue in turns.dialogue}
            for dialogue, dial_turn_id, dial_turn_text in zip(
                turns.dialogue, turns.turn_id, turns.text
            ):
                dial_to_turns[dialogue].append((dial_turn_id, dial_turn_text))
        with ez.Timer('Mapping turns to contexts'):
            turn_to_context = {}
            for dialogue, dial_turns in dial_to_turns.items():
                context = []
                for dial_turn_id, dial_turn_text in dial_turns:
                    context.append(dial_turn_text)
                    context_lines = list(reversed(list(zip(it.cycle('AB'), reversed(context)))))
                    turn_to_context[dial_turn_id] = '\n'.join([f"{speaker}: {utt}" for speaker, utt in context_lines])
        if for_training and self.hyperparameters.neg_examples_ratio:
            with ez.Timer('Adding negatives for training'):
                turn_targets_for_negs = {domain: [] for domain in domain_to_slots}
                for domain, turn_id, spkr in zip(turns.domain, turns.turn_id, turns.speaker):
                    if (
                        domain in domain_to_slots
                        and (not self.hyperparameters.exclude_speakers
                             or spkr not in self.hyperparameters.exclude_speakers)
                    ):
                        turn_targets_for_negs[domain].append(turn_id)
                num_training_examples = len(data.predictions)
                num_negative_examples = int(num_training_examples * self.hyperparameters.neg_examples_ratio)
                slots_to_domain = {
                    slot_id: domain for slot_id, domain in zip(data.slots.slot_id, data.slots.domain)
                }
                training_slots = list(slots_to_domain)
                slot_id_to_name = {sid: name for sid, name in zip(data.slots.slot_id, data.slots.slot)}
                domains = list(turn_targets_for_negs)
                negatives_slots = rng.sample(training_slots, num_negative_examples)
                negatives_to_add = []
                for slot_id in tqdm(negatives_slots, desc='Negatives...'):
                    neg_slot_name = slot_id_to_name[slot_id]
                    domain = slots_to_domain[slot_id]
                    retries = 1_000
                    while (neg_domain := rng.choice(domains)) == domain or neg_slot_name in domain_to_slots[neg_domain]:
                        if not retries:
                            print('Warning: could not find a domain to add a negative example for', neg_slot_name)
                            break
                        retries -= 1
                        continue
                    neg_turn_candidates = turn_targets_for_negs[neg_domain]
                    if not neg_turn_candidates:
                        print('Warning: could not find a turn to add a negative example for',
                            neg_slot_name, 'in domain', domain)
                        continue
                    neg_turn_id = rng.choice(neg_turn_candidates)
                    negative_example = (neg_slot_name, None, neg_turn_id, slot_id, None)
                    negatives_to_add.append(negative_example)
                negative_values = dst.SlotValue.of(negatives_to_add, fill=None)
                negative_preds = dst.Prediction.of(dict(
                    slot_value_id=list(negative_values.slot_value_id),
                    slot_id=list(negative_values.slot_id)
                ), fill=None)
                values += negative_values
                predictions += negative_preds
        with ez.Timer('Creating prompts'):
            examples = predictions.slot_value_id << values.slot_value_id
            slots_to_descriptions = {
                slot_id: desc for slot_id, desc in zip(slots.slot_id, slots.description)
            }
            slots_to_candidates = {slot_id: set() for slot_id in slots.slot_id}
            turn_slot_to_value = {(turn_id, slot_id): value for turn_id, slot_id, value in zip(
                values.turn_id, values.slot_id, values.value
            )}
            for slot_id, candidate_value in zip(candidates.slot_id, candidates.candidate_value):
                candidate_values = ez.split(candidate_value, '|', '<', '>', ', ')
                slots_to_candidates[slot_id].update(
                    c for c in candidate_values if len(c) < self.hyperparameters.max_candidate_value_string_length
                )
            prompt_infos = []
            ex_is_positives = []
            ex_has_description = []
            ex_candidate_count = []
            for ex_turn_id, ex_slot, ex_slot_id in list(zip(
                examples.turn_id, examples.slot, examples.slot_id
            )):
                ex_context = turn_to_context[ex_turn_id]
                ex_description = slots_to_descriptions[ex_slot_id]
                ex_candidates = slots_to_candidates[ex_slot_id]
                ex_value = turn_slot_to_value.get((ex_turn_id, ex_slot_id))
                if for_training and ex_value and self.hyperparameters.train_filters_out_descriptions_with_actual_value:
                    if ex_description and len(ex_value) > 3 and ex_value in ex_description:
                        ex_description = None
                prompt_infos.append((ex_context, ex_slot, ex_description, ex_candidates, ex_value))
                ex_is_positives.append(ex_value is not None and ex_value != self.hyperparameters.request_token)
                ex_has_description.append(bool(ex_description))
                ex_candidate_count.append(len(ex_candidates))
            with_descriptions = {i for i in range(len(prompt_infos)) if ex_has_description[i]}
            with_description_only = set()
            with_categories = {i for i in range(len(prompt_infos))
                if 1 < ex_candidate_count[i] <= self.hyperparameters.is_categorical_limit
            }
            with_candidates = {i for i in range(len(prompt_infos)) if ex_candidate_count[i]}
            with_value = {
                i for i in range(len(prompt_infos)) if ex_is_positives[i] and i in with_candidates
            }
            h = self.hyperparameters
            if for_training:
                num_with_descriptions = int(
                    len(prompt_infos) * h.train_percent_with_description
                )
                num_description_only = int(
                    num_with_descriptions * h.train_percent_description_only
                )
                num_with_categories = int(
                    len(prompt_infos) * h.train_percent_with_categories
                )
                num_with_candidates = max(0, int(
                    len(prompt_infos) * h.train_percent_with_value_exs
                ))
                num_value_ex_includes_actual_value = int(
                    num_with_candidates * h.train_percent_value_ex_includes_actual_value
                ) if h.train_percent_value_ex_includes_actual_value is not None else None
                with_descriptions = set(rng.sample(
                    list(with_descriptions), min(num_with_descriptions, len(with_descriptions))
                ))
                with_description_only = set(rng.sample(
                    list(with_descriptions), min(num_description_only, len(with_descriptions))
                ))
                with_categories = set(rng.sample(
                    list(with_categories), min(num_with_categories, len(with_categories))
                ))
                with_candidates -= with_categories
                with_candidates = set(rng.sample(
                    list(with_candidates), min(num_with_candidates, len(with_candidates))
                ))
                with_value &= with_candidates
                with_value = set(rng.sample(
                    list(with_value), min(num_value_ex_includes_actual_value, len(with_value))
                )) if num_value_ex_includes_actual_value is not None else None
            else:
                with_value = set()
                if not self.hyperparameters.eval_with_categories:
                    with_categories = set()
                if not self.hyperparameters.eval_with_value_exs:
                    with_candidates = set()
            numbers = list(range(1, len(self.hyperparameters.train_num_candidates_distribution)+1))
            template = '''{context}\n\n{prompt}\n{slot}{description}{examples}'''
            prompts = []
            for i, (
                ex_context, ex_slot, ex_description, ex_candidates, ex_value
            ) in enumerate(prompt_infos):
                if i in with_description_only:
                    p_slot = ''
                    p_description = ex_description.rstrip('.')
                elif i in with_descriptions:
                    p_slot = '' + ex_slot
                    p_description = ': ' + ex_description.rstrip('.')
                else:
                    p_slot = '' + ex_slot
                    p_description = ''
                if (
                    i in with_categories or
                    (with_value and i in with_value)
                    or i in with_candidates
                ):
                    ex_value = ez.split(ex_value, ', ', '|', '<', '>') if ex_value else [ex_value]
                    if for_training:
                        if with_value is not None:
                            if i in with_categories or i in with_value:
                                example_values = ex_candidates | {*ex_value} - {None, h.request_token}
                            else:
                                example_values = ex_candidates - {*ex_value, None, h.request_token}
                        else:
                            example_values = ex_candidates - {None, h.request_token}
                    else:
                        if i in with_categories:
                            example_values = ex_candidates
                        else:
                            example_values = ex_candidates - {None, h.request_token}
                    if i in with_candidates and i not in with_categories:
                        num_candidates, = rng.choices(
                            numbers, self.hyperparameters.train_num_candidates_distribution
                        )
                        example_values = rng.sample(
                            example_values, min(len(example_values), num_candidates))
                        example_values = [x.strip() for x in example_values if x.strip()]
                        p_examples = ' (e.g. ' + ', '.join(example_values) + ')?' if example_values else '?'
                    else:
                        p_examples = ' [' + ', '.join(example_values) + ']?'
                else:
                    p_examples = '?'
                prompts.append(template.format(
                    prompt=self.hyperparameters.slot_instruction_prefix,
                    context=ex_context, slot=p_slot, description=p_description, examples=p_examples
                ))
        predictions.prompt[:] = prompts

    def postprocess(self, data: dst.Data):
        value_map = {
            self.hyperparameters.empty_slot_token: None
        }
        predictions = [
            value_map.get(x.strip(), x.strip()) for x in data.predictions.generated
        ]
        if self.hyperparameters.prediction_lowercase:
            predictions = [
                x.lower() if isinstance(x, str) else x for x in predictions
            ]
        for i, prediction in enumerate(predictions):
            if not prediction:
                continue
            timecolon = prediction.find(':')
            if timecolon >= 0 and prediction[timecolon-1: timecolon+3].replace(':', '').isnumeric():
                if len(prediction[:timecolon]) == 1:
                    predictions[i] = '0'+prediction
        if self.hyperparameters.prediction_fuzzy_match_candidates:
            slots_to_candidates = {slot: [] for slot in data.slots.slot_id}
            for slot, candidate in zip(data.value_candidates.slot_id, data.value_candidates.candidate_value):
                slots_to_candidates[slot].append(candidate)
            slot_values_to_slots = {
                svid: slot for svid, slot in zip(data.slot_values.slot_value_id, data.slot_values.slot_id)
            }
            for i, (svid, prediction) in enumerate(zip(data.predictions.slot_value_id, predictions)):
                if prediction:
                    candidates = slots_to_candidates.get(slot_values_to_slots[svid])
                    if candidates:
                        matches = dl.get_close_matches(prediction, candidates)
                        if matches:
                            predictions[i] = matches[0]
        data.predictions.prediction[:] = predictions


    def training(self, data: dst.Data, yield_every_x_epochs=1):
        self.preprocess(data, for_training=True)
        examples = data.predictions.slot_value_id << data.slot_values.slot_value_id
        prompts = list(examples.prompt)
        values = [v or self.hyperparameters.empty_slot_token for v in examples.value]
        display = [f"{p} -> {v}" for p, v in zip(prompts, values)]
        display = rng.sample(display, 100)
        print('\n\n'.join(display))
        return self._training(prompts, values, yield_every_x_epochs)
    def _training(self, prompts, values, yield_every_x_epochs=1):
        for perplexity in self.model.training(prompts, values, yield_every_x_epochs):
            yield perplexity

    def train(self, data: dst.Data) -> list[float]:
        return list(self.training(data))

    def perplexity(self, data: dst.Data) -> float:
        self.preprocess(data)
        examples = data.predictions.slot_value_id << data.slot_values.slot_value_id
        prompts = list(examples.prompt)
        values = [v or self.hyperparameters.empty_slot_token for v in examples.value]
        ppl = self.model.perplexity(prompts, values)
        return ppl

    def predict(self, data: dst.Data):
        if not any(data.predictions.prompt):
            self.preprocess(data)
        prompts = list(data.predictions.prompt)
        generated = self.model.generate(prompts)
        data.predictions.generated[:] = generated
        self.postprocess(data)
        return data


def main():
    pass


if __name__ == '__main__':
    main()

    th = LlamaTrackerHyperparameters(uncased=True)
    th.settings = dict(uncased=True)

    tracker = LlamaTracker(th)
    tracker.train(...)
    tracker.save('blah')

    #####

    LlamaTracker(LlamaTrackerHyperparameters(model_path='blah', repetition_penalty=2.0))

