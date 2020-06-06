import numpy as np
from sklearn.model_selection import train_test_split
from collections import UserDict


def conv2variable(arr, name, dtype=None):
    varr = np.array(arr).T
    dtype = (dtype if dtype else type(varr[0]))
    var = VariableWrapper(name=name, data=arr, dtype=dtype)
    return var


class VariableWrapper(UserDict):

    def __init__(self, name, data, _type='continuous', dtype=np.float):
        super().__init__({"data": data, 'var_type': _type, 'dtype': dtype})
        self.__name__ = name

    @property
    def name(self): return self.__name__

    @name.setter
    def name(self, name): self.__name__ = name

    @property
    def data(self):
        return self['data']

    @data.setter
    def data(self, v):
        self['data'] = v

    def __len__(self):
        return len(self['data'])


class VariablesContainer(UserDict):

    def __init__(self, **items):
        super().__init__(items)

    def get_variable(self, name):
        return self.get(name, None)

    def remove_variable(self, name):
        return self.pop(name, None)

    def set_names(self, *names, idx=0):

        if type(names[0]) is tuple:
            idx, name = names[0]
        else:
            name = names[0]

        try:
            var = self.pop(idx)
        except KeyError:
            raise KeyError("Invalid key, {name[0][0]} not found.")
        else:
            var.name = name
            self.update({var.name: var})

        if len(names) == 0:
            print("New names set.")
        else:
            self.set_names(*names, idx=idx + 1)

    def to_array(self):  # TODO: Issue with multiple samples

        return [list(z) for z in
                zip(*[self[a]['data'].T for a in self])
                ]

    def get_data(self, name):
        return self[name]['data']


class Dataset(object):

    def __init__(self, X=None, y=None, random_seed=None, x_opts=None,
                 y_opts=None):
        self._X = VariablesContainer()
        self._y = VariablesContainer()
        x_opts = ({} if not x_opts else x_opts)
        y_opts = ({} if not y_opts else y_opts)

        self._num_rows = 0
        self._num_features, self._num_targets = 0, 0
        self._x_train, self._y_train = None, None
        self._x_test, self._y_test = None, None
        self._x_valid, self._y_valid = None, None
        if X is not None:
            self.set_X(vals=X, names=x_opts.get("names", None),
                       dtypes=x_opts.get("dtypes", None))
        if y is not None:
            self.set_y(vals=y, names=y_opts.get("names", None),
                       dtypes=y_opts.get("dtypes", None))

        self._randstate = np.random.RandomState(random_seed)

    @property
    def x_train(self):
        return self._x_train

    @property
    def y_train(self):
        return self._y_train

    @property
    def x_test(self):
        return self._x_test

    @property
    def y_test(self):
        return self._y_test

    @property
    def x_valid(self):
        return self._x_valid

    @property
    def y_valid(self):
        return self._y_valid

    @property
    def num_rows(self):
        return self._num_rows

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def num_features(self):
        return self._num_features

    @property
    def num_targets(self):
        return self._num_targets

    def get_variables(self, varset, *names):
        v = (self._X if varset == 'x' else self._y)
        for name in names:
            yield v.get_variable(name)

    def get_sample(self, i):
        return

    @staticmethod
    def _norm_valid_size(train_sz, valid_sz):
        return round(float(valid_sz / train_sz), 2)

    def train_test_split(self, train_sz, test_sz=None, valid_sz=None,
                         random_state=None, stratify=None, shuffle=True,
                         X=None, y=None):  # TODO: Issue with multiple samples

        X = (self.get_X() if X is None else X)
        y = (self.get_y() if y is None else y)
        rand = (self._randstate if not random_state else random_state)
        opts = {"random_state": rand, 'stratify': stratify, "shuffle": shuffle}
        test_sz = (test_sz if test_sz is not None and train_sz + test_sz <= 1.
                   else round(1. - train_sz, 2))
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=test_sz,
                                                            train_size=train_sz,
                                                            **opts)
        if train_sz + test_sz == 1.:
            X_val, y_val = None, None
        else:

            v_sz = (
                0. if not valid_sz or (train_sz + test_sz == 1.) else valid_sz)
            v_sz = Dataset._norm_valid_size(train_sz=1 - (test_sz + valid_sz),
                                            valid_sz=v_sz)

            tr_sz = round(1. - v_sz, 2)
            v_sz = round(v_sz, 2)
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                              train_size=tr_sz,
                                                              test_size=v_sz,
                                                              **opts)

        self._x_train, self._y_train = X_train, y_train
        self._x_test, self._y_test = X_test, y_test
        self._x_valid, self._y_valid = X_val, y_val

    def get_X(self):  # TODO: Issue with multiple samples
        arr = []
        for x in self._X:
            arr.extend(x.to_array())
        return arr

    def get_y(self):  # TODO: Issue with multiple samples
        arr = []
        for y in self._y:
            arr.extend(y.to_array())
        return arr

    def _bld_contains__(self, vals_arr, names=None, dtypes=None):

        vars_container = VariablesContainer()
        for vidx in range(len(vals_arr)):
            if names:
                name = names[vidx]
            else:
                name = vidx
            if dtypes:
                dtype = dtypes[vidx]
            else:
                dtype = type(vals_arr[vidx][0])
            d = {name: conv2variable(arr=vals_arr[vidx], name=name,
                                     dtype=dtype)}
            vars_container.update(d)
        return vars_container

    def set_X(self, vals, names=None, dtypes=None):
        vals_arr = np.asarray(vals)

        if len(vals_arr.shape) == 2:
            n = vals_arr.shape[0]
            nfeats = vals_arr.shape[1]
            self._X = self._bld_contains__(vals_arr=vals_arr.T,
                                           names=names, dtypes=dtypes)
        elif len(vals_arr.shape) == 1:
            n = len(vals_arr)
            nfeats = 1
            d = {names: conv2variable(arr=vals_arr.T,
                                      name=names, dtype=dtypes)}
            self._X.update(d)
        else:
            raise ValueError("Value array cannot have more that 3 dimensions")
        self._num_rows = n
        self._num_features = nfeats

    def set_y(self, vals, names=None, dtypes=None):
        vals_arr = np.asarray(vals)
        if len(vals_arr.shape) == 2:
            n, ntargs = vals_arr.shape
            self._y = self._bld_contains__(vals_arr=vals_arr.T,
                                           names=names, dtypes=dtypes)
        elif len(vals_arr.shape) == 1:

            n = len(vals_arr)
            ntargs = 1
            d = {names: conv2variable(arr=vals_arr.T,
                                      name=names, dtype=dtypes)}
            self._y.update(d)
        self._num_rows = n
        self._num_targets = ntargs

    def add_variable(self, data, varset, name=None, dtype=np.float,
                     _type='continuous'):
        v = (self._X if varset.lower() == 'x' else self._y)
        d = np.asarray(data).T
        if self._num_rows != len(data):
            raise IndexError("data length does not match existing")
        else:
            nm = (len(v[0]) if not name else name)
            var = VariableWrapper(name=nm, data=d, dtype=dtype, _type=_type)
            v.update({var.name: var})
            if varset.lower() == 'x':
                self._num_features += (
                    1 if len(data.shape) == 1 else data.shape[-1])
            else:
                self._num_targets += (
                    1 if len(data.shape) == 1 else data.shape[-1])

    def add_dummy(self, varset, groups, data=None, name=None, dtype=np.int,
                  _type='dummy'):
        if data is None:
            data = self._randstate.choice(groups, size=self.num_rows)
        self.add_variable(data=data, varset=varset, name=name, dtype=dtype,
                          _type='discrete')

    def apply_transformer(self, varset, name, fn=None, *args, **kwargs):
        vv = (self._X if varset.lower() == 'x' else self._y)
        varvalues = vv.get_data(name)

        if fn is None:
            op = kwargs.pop("operator")
            mult = kwargs.pop("multiplier")
            if op is None or mult is None:
                raise ValueError(
                    "if function is None must specify operator function"
                    "and mulitplier value")
            elif op not in ["*", "**", "+", "-", "/", "//", "^", "floor",
                            "x", "mod", "%", "modulo"]:
                raise ValueError("Invalid operator")
            elif type(mult) is not int or type(mult) is not float:
                raise TypeError("Invalid multiplier, must be int or float")
            else:
                vals = self._builtin_transform_func__(vals=varvalues,
                                                      multiplier=mult, op=op)
        else:
            try:
                vals = fn.__call__(*args, **kwargs)
            except Exception as err:
                raise Exception("func must be callable\n" + err)

        vr = vv.get_variable(name=name)
        vr.data = vals
        print("Variable {} - updated.".format(name))

    def _builtin_transform_func__(self, vals, multiplier, op):
        def func(val):
            if op == '*' or op == 'x':
                v = val * multiplier
            elif op == '+':
                v = val + multiplier
            elif op == "-":
                v = val - multiplier
            elif op == "**" or op == "^":
                v = val ** multiplier
            elif op == "/":
                v = val / multiplier
            elif op == '//' or op == 'floor':
                v = val // multiplier
            elif op == '%' or op == "mod" or op == 'modulo':
                v = val % multiplier
            else:
                raise ValueError("Invalid operator. consult docs for list of "
                                 "accepted operators")
            return v

        return list(map(func, vals))
