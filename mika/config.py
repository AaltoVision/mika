import argparse
import sys
import yaml
import itertools
from copy import deepcopy


class Parsers:
    @staticmethod
    def boolean(v):
        if v.lower() in ['yes', 'true', '1']:
            return True
        elif v.lower() in ['no', 'false', '0']:
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    @staticmethod
    def eval(v):
        assert type(v) == str
        return eval(v)


class Argument:

    def __init__(self, dtype, value=None, descr=None):
        self._dtype = dtype
        self._parser = dtype
        if type(value) == list:
            self._value = value
        else:
            self._value = [value]
        self._descr = descr

        if dtype == tuple:
            self._parser = Parsers.eval
        elif dtype == bool:
            self._parser = Parsers.boolean

        self._nargs = '+'

    @property
    def value(self):
        if type(self._value) == list:
            return self._value[0]

        return self._value

    @value.setter
    def value(self, value):
        if type(value) == list:
            self._value = value
        else:
            self._value = [value]

    @property
    def dtype(self):
        return self._dtype

    @property
    def default(self):
        return self._default

    @property
    def descr(self):
        return self._descr

    @property
    def nargs(self):
        return self._nargs

    def __str__(self):
        return '{}, -- {}'.format(self.value, self.descr)


class Config(object):
    def __init__(self, **kwargs):
        self.params = kwargs
        self._name = None
        self._parent = None
        self._update_attrs()
        self._update_arg_prefix()

    @property
    def parent(self):
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value
        self._update_arg_prefix()

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self._update_arg_prefix()

    def _update_arg_prefix(self):
        self.arg_prefix = '{}{}'.format('' if self.parent == None else self.parent.arg_prefix,
                                        '' if self._name == None else self._name + '.')
        for _, v in self.params.items():
            if type(v) == Config:
                v._update_arg_prefix()

    def parse_arguments(self, args):
        parser = argparse.ArgumentParser()
        for k,v in self.params.items():
            if type(v) == type(self):
                v.parse_arguments(args)
            else:
                parser.add_argument('--{}{}'.format(self.arg_prefix, k),
                                    type=v._parser, default=v.value, dest=k, nargs=v.nargs)

        parsed, unknown = parser.parse_known_args(args)
        for k in unknown:
            if k == '{}'.format('--{}'.format(self.arg_prefix)):
                print('Unknown argument: {}'.format(k))

        for k,v in vars(parsed).items():
            if k in self.params and '--{}{}'.format(self.arg_prefix, k) in args:
                self.params[k].value = v
                if type(v) == list:
                    setattr(self, k, v[0])
                else:
                    setattr(self, k, v)

        return parsed


    def help(self):
        r = ''
        for k,v in self.params.items():
            if type(v) == type(self):
                r += '{}\n'.format(v.help())
            else:
                r += '{}{}: {}\n'.format(self.arg_prefix, k, v)

        print(r)

    def __str__(self):
        n_parents = 0
        p = self.parent
        while p is not None:
            n_parents += 1
            p = p.parent

        r = ''
        if n_parents == 0:
            r = 'Configuration:'

        r += '\n'
        for k,v in self.params.items():
            r += ('\t' * (n_parents+1)) + '{}: {}\n'.format(k, v)

        return r

    def __dict__(self):
        return {k: v.__dict__() if type(v) == Config else v.value for k,v in self.params.items()}

    def load_from_dict(self, items):
        for k, v in items.items():
            if k in self.params:
                if type(v) == dict and type(self.params[k]) == type(self):
                    self.params[k].load_from_dict(v)
                else:
                    self.params[k].value = v
                    setattr(self, k, v)

        return self

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            f.write(yaml.dump(self.__dict__(), default_flow_style=False))

    def load_from_file(self, filename):
        with open(filename, 'r') as f:
            items = yaml.load(f)
            self.load_from_dict(items)

    def get_all_args(self):
        args = []
        for p in self.params.items():
            if type(p[1]) == Config:
                args += p[1].get_all_args()
            else:
                args += [(p[0], p[1]._value)]

        return args

    def _update_attrs(self):
        for k, v in self.params.items():
            if type(v) == type(self):
                v._name = k
                v._parent = self
                v._update_attrs()
                setattr(self, k, v)
            else:
                setattr(self, k, v.value)

    def flatten(self):
        _dict = self.__dict__()
        l = []
        for k, v in self.params.items():
            if type(v) == Config:
                l += v.flatten()
            else:
                l += [('{}{}'.format(self.arg_prefix, k), v._value)]

        return l

    def __deepcopy__(self, memo):
        c = type(self)()
        for k,v in self.params.items():
            if type(v) == type(self):
                c.params[k] = deepcopy(v, memo)
            else:
                c.params[k] = Argument(v._dtype, v._value, v._descr)

        c._update_attrs()
        return c


def cartesian(config):
    flat = config.flatten()
    keys = [x[0] for x in flat]
    values = [x[1] for x in flat]
    print('Parameter sets:')
    [print(k,v) if len(v) > 1 else None for k,v in flat]
    cartesian = list(itertools.product(*values))
    lists = [list(zip(keys, v)) for v in cartesian]
    def _reconstruct(x):
        _dict = {}
        for k, v in x:
            keys = k.split('.')
            _d = _dict
            for i in range(len(keys) - 1):
                if keys[i] not in _d.keys():
                    _d[keys[i]] = {}
                _d = _d[keys[i]]

            _d[keys[-1]] = v
        return _dict

    reconstructed = [_reconstruct(l) for l in lists]
    configs = [deepcopy(config).load_from_dict(d) for d in reconstructed]
    return configs

if __name__ == '__main__':
    arg = Argument
    config = Config(actor=Config(lr=arg(float, 1e-3, 'LR')),
                    critic=Config(test=Config(
                                              testi=arg(str, 'Moi', 'Moikka'),
                                              lr=arg(float, 1e-4, 'Critic LR')
                                              )),
                    batch_size=arg(int, 32, 'batch size'),
                    test_tuple=arg(tuple, (1, 2), 'Test tuple'))

    args = config.parse_arguments(sys.argv)
    import pdb;pdb.set_trace()
    configs = cartesian(config)
    for c in configs:
        print(c)

