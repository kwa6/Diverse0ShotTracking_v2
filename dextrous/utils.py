
import ezpyzy as ez
import matplotlib.pyplot as plt
import io
import fabric as fab
import paramiko
import pathlib as pl
import itertools as it
import dataclasses as dc
import dextrous.dst_data as dst
import random as rng

hosts = dict(
    tebuna='localhost:55555',
    h100='localhost:55556'
)

user = 'jdfinch'
project = pl.Path('/local/scratch/jdfinch/dextrous/')

@dc.dataclass
class Ssh:
    user: str
    password_file: str
    host: str
    port: int = None
    project: str = None

    def __post_init__(self):
        self.password = ez.File(self.password_file).read().strip()
        if ':' in self.host:
            self.host, self.port = self.host.split(':')
            self.port = int(self.port)
        self.port = self.port or 22


def download(machine, path):
    password = ez.File('~/.pw/emory').read().strip()
    credentials = dict(
        user=user,
        connect_kwargs=dict(password=password)
    )
    with fab.Connection(hosts[machine], **credentials) as conn:
        path = pl.Path(path)
        is_folder = conn.run(
            f'test -d {project/path} && echo 1 || echo 0'
        ).stdout.strip() == '1'
        if is_folder:
            tar_file = path.with_suffix('.tar.gz')
            conn.run(f'cd {project} && tar -czvf {tar_file} {path}')
            conn.get(f'{project / tar_file}', str(tar_file))
            conn.run(f'rm {project / tar_file}')
            conn.local(f'tar -xzvf {tar_file}')
            conn.local(f'rm {tar_file}')
        else:
            conn.get(f'{project/path}', str(path))

def transfer_remote2remote(from_machine, from_path, to_machine, to_path=None, is_folder=False):
    if to_path is None:
        to_path = from_path
    if not str(from_path).startswith('/'):
        from_path = project / from_path
    if not str(to_path).startswith('/'):
        to_path = project / to_path
    password = ez.File('~/.pw/emory').read().strip()

    hub_machine = hosts['tebuna']
    if ':' in hub_machine:
        hub_machine, hub_port = hub_machine.split(':')
        hub_port = int(hub_port)
    else:
        hub_port = 22

    args = ''
    if is_folder:
        args = '-r'

    if from_machine == 'h100' and to_machine == 'tebuna':
        command = f'sshpass -p {password} scp {args} {user}@{from_machine}:{from_path} {to_path}'
    elif from_machine == 'tebuna' and to_machine == 'h100':
        command = f'sshpass -p {password} scp {args} {from_path} {user}@{to_machine}:{to_path}'
    else:
        raise ValueError(f'Cannot transfer from {from_machine} to {to_machine}')

    hub = paramiko.SSHClient()
    hub.load_system_host_keys()
    hub.connect(hub_machine, port=hub_port, username=user, password=password)
    stdin, stdout, stderr = hub.exec_command(command)
    for line in stdout.readlines():
        print(line)
    for line in stderr.readlines():
        print(line)
    hub.close()



def roundrobin(*iterables):
    """Yields an item from each iterable, alternating between them.
        >>> list(roundrobin('ABC', 'D', 'EF'))
        ['A', 'D', 'E', 'B', 'F', 'C']
    """
    pending = len(iterables)
    nexts = it.cycle(iter(items).__next__ for items in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = it.cycle(it.islice(nexts, pending))


def create_line_graph_image(**axes):
    ((xa, xs), (ya, ys)) = axes.items()
    plt.plot(xs, ys)
    plt.xlabel(xa)
    plt.ylabel(ya)
    plt.yscale('log')
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return buffer.getvalue()


def plot_lines(**performances):
    """
    Plot lines of performance over time for different models.

    Args:
    performances: Keyword arguments where the key is the model name and the value is a list of floats representing
                  the performance of that model over time.
    """
    fig, ax = plt.subplots()
    for model, data in performances.items():
        ax.plot(range(len(data)), data, label=model)
    ax.set_xlabel('Time')
    ax.set_ylabel('Performance')
    ax.set_title('Performance of Models Over Time')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.subplots_adjust(right=0.6)
    plt.show()


if __name__ == '__main__':
    a = []
    b = [1, 2, 3, 4, 5]
    c = [6, 7, 8]
    d = [9, 10, 11, 12]
    e = [13, 14, 15, 16, 17]
    f = [18, 19]
    print(list(roundrobin(a, b, c, d, e, f)))



