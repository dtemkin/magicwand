import numpy as np
from sklearn.model_selection import train_test_split
from collections import UserList, UserDict


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

    def __len__(self):
        return len(self['data'])


class VariablesContainer(UserDict):

    def __init__(self, **items):
        super().__init__(items)

    def set_names(self, *names, idx=0):

        if type(names[0]) is tuple:
            idx, name = names.pop(0)
        else:
            name = names.pop(0)

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
            self.set_names(*names, idx=idx+1)

    def to_array(self):

        return np.array([list(z) for z in
                         zip(*[self[a]['data'].T for a in self])
                         ])

    def get_data(self, name):
        return self[name]['data']


class Dataset(object):

    def __init__(self, X=None, y=None, random_seed=None, x_opts=None, y_opts=None):
        self._X = []
        self._y = []
        x_opts = ({} if not x_opts else x_opts)
        y_opts = ({} if not y_opts else y_opts)
        self._num_rows, self._num_vars = 0, 0
        self._num_samples, self._num_targets = 0, 0

        if X:
            self.set_X(vals=X, names=x_opts.get("names", None),
                       dtypes=x_opts.get("dtypes", None))
        if y:
            self.set_y(vals=y, names=y_opts.get("names", None),
                       dtypes=y_opts.get("dtypes", None))

        self._randstate = np.random.RandomState(random_seed)

    def get_sample(self, i):
        return self._X[i]

    def train_test_split(self, test_sz, valid_sz=None,
                         random_state=None, stratify=None, shuffle=True):
        rand = (self._randstate if not random_state else random_state)
        opts = {"random_state": rand, 'stratify': stratify, "shuffle": shuffle}

        X_train, X_test, y_train, y_test = train_test_split(self.get_X(),
                                                            self.get_y(),
                                                            test_sz=test_sz,
                                                            **opts)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
                                                          test_sz=valid_sz,
                                                          **opts)

        setattr(self, 'x_train', X_train)
        setattr(self, 'y_train', y_train)
        setattr(self, 'x_test', X_test)
        setattr(self, 'y_test', y_test)
        setattr(self, 'x_valid', X_val)
        setattr(self, 'y_valid', y_val)

    def get_X(self):
        if len(self._X) == 1:
            yield self._X[0].to_array()
        else:
            for x in self._X:
                yield x.to_array()

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

        if len(vals_arr.shape) == 3:
            samples, n, nfeats = vals_arr.shape
            for sample_id in range(len(samples)):
                self._X.append(self._bld_contains__(vals_arr=vals_arr[sample_id].T,
                                                    names=names, dtypes=dtypes))

        elif len(vals_arr.shape) == 2:
            n = vals_arr.shape[0]
            nfeats = vals_arr.shape[1]
            samples = 1
            self._X.append(self._bld_contains__(vals_arr=vals_arr.T,
                                                names=names, dtypes=dtypes))
        elif len(vals_arr.shape) == 1:
            n = len(vals_arr)
            nfeats = 1
            samples = 1

            vars_container = VariablesContainer()
            d = {names: conv2variable(arr=vals_arr.T,
                                      name=names, dtype=dtypes)}
            vars_container.update(d)
            self._X.append(vars_container)
        else:
            raise ValueError("Value array cannot have more that 3 dimensions")
        self._num_rows = n
        self._num_samples = samples
        self._num_features = nfeats

    def get_y(self):
        if len(self._y) == 1:
            yield self._y[0].to_array()
        else:
            for y in self._y:
                yield y.to_array()

    def set_y(self, vals, names=None, dtypes=None):
        vals_arr = np.asarray(vals)
        if len(vals_arr.shape) == 3:
            samples, n, ntargs = vals_arr.shape
            for sample_id in range(len(samples)):
                self._y.append(self._bld_contains__(vals_arr=vals_arr[sample_id].T,
                                                    names=names,
                                                    dtypes=dtypes))

        elif len(vals_arr.shape) == 2:
            n, ntargs = vals_arr.shape
            samples = 1
            self._y.append(self._bld_contains__(vals_arr=vals_arr.T,
                                                names=names, dtypes=dtypes))
        else:
            vars_container = VariablesContainer()
            n = len(vals_arr)
            ntargs = 1
            samples = 1
            d = {names: conv2variable(arr=vals_arr.T,
                                      name=names, dtype=dtypes)}
            vars_container.update(d)
            self._y.append(vars_container)
        self._num_rows = n
        self._num_samples = samples
        self._num_targets = ntargs

    def add_variable(self, data, var, name=None, dtype=np.float,
                     _type='continuous'):

        v = (self._X if var.lower() == 'x' else self._y)
        d = np.asarray(data).T
        if self._num_rows != len(data):
            raise IndexError("data length does not match existing")
        else:
            self._num_vars += (1 if len(data.shape) == 1 else data.shape[-1])
            self._num_samples += (0 if len(data.shape) < 3 else data.shape[0])
            nm = (len(v[0]) if not name else name)
            var = VariableWrapper(name=nm, data=d, dtype=dtype, _type=_type)
            for samp in v:
                samp.update({var.name: var})

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


#    def add_y_variable(self, name, data, dtype=np.float, _type="continuous"):
#        var = VariableWrapper(name=name, data=data, dtype=dtype, _type=_type)
#        self._y.update({var.__name__: var})
#        mat = [list(z) for z in zip(*self._y.get('matrix'), *var.data)]
#        self._y.update({"matrix": mat})
    # def _apply_trans__(self, func, data, *args, **kwargs):
    #     try:
    #         t = func.__call__(data, *args, **kwargs)
    #     except Exception as err:
    #         raise Exception(f"there was an error in applying the "
    #                         f"transformer, {err}")
    #     else:
    #         return t
    #
    # def _transform__(self, _type, transformer, in_place,
    #                  var, axis, *args, **kwargs):
    #     """
    #     :param transformer: transformation function, must be callable,
    #     must accept 2d array as 1st argument, remaining arguments passed in args,
    #     or kwargs.
    #     :param var: apply transformation to specific columns or rows,
    #     (Default: None - Apply to all)
    #     :param axis: apply transformation along specified axis
    #     (0: row-wise, 1: col-wise)
    #     :return: array if 'in-place' == False else None
    #     """
    #     if axis == 0:
    #         data = (self.X if _type == 'x' else self.y)
    #
    #     elif axis == 1:
    #         data = (self.X.T if _type == 'x' else self.y.T)
    #
    #     else:
    #         raise Exception("Could not apply transformation, "
    #                         "invalid axis value {axis}")
    #
    #     if var is None:
    #         var = range(len(data))
    #     elif type(var) == int:
    #         var = [var]
    #     elif type(var) == list and all([type(v)==int for v in var]):
    #         pass
    #     else:
    #         raise ValueError("Invalid vars argument must be (None, int or "
    #                          "list of int)")
    #
    #     transformed = []
    #     for d in range(len(data)):
    #         if d in var:
    #             t = self._apply_trans__(transformer, d, *args, **kwargs)
    #             if in_place:
    #                 if _type == 'x' and axis == 0:
    #                     self.X[d] = t
    #                 elif _type == 'x' and axis == 1:
    #                     self._X.T[d] = t
    #                 elif _type == 'y' and axis == 0:
    #                     self._y[d] = t
    #                 elif _type == 'y' and axis == 1:
    #                     self._y.T[d] = t
    #             else:
    #                 transformed.append(t)
    #
    #     return (None if in_place else transformed)
    #
    # def x_transform(self, transformer, in_place=True,
    #                 var=None, axis=0, *args, **kwargs):
    #     """
    #     :param transformer: transformation function, must be callable,
    #     must accept 2d array as 1st argument, remaining arguments passed in args,
    #     or kwargs.
    #     :param var: apply transformation to specific columns or rows,
    #     (Default: None - Apply to all)
    #     :param axis: apply transformation along specified axis
    #     (0: row-wise, 1: col-wise)
    #     :return: array if 'in-place' == False else None
    #     """
    #     return self._transform__(_type='x', transformer=transformer,
    #                              in_place=in_place, var=var, axis=axis,
    #                              args=args, kwargs=kwargs)
    #
    # def y_transform(self, transformer, in_place=True,
    #                 var=None, axis=0, *args, **kwargs):
    #     """
    #     :param transformer: transformation function, must be callable,
    #     must accept 2d array as 1st argument, remaining arguments passed in args,
    #     or kwargs.
    #     :param var: apply transformation to specific columns or rows,
    #     (Default: None - Apply to all)
    #     :param axis: apply transformation along specified axis
    #     (0: row-wise, 1: col-wise)
    #     :return: array if 'in-place' == False else None
    #     """
    #     return self._transform__(_type='y', transformer=transformer,
    #                              in_place=in_place, var=var, axis=axis,
    #                              args=args, kwargs=kwargs)