# Copyright 2022 Cisco Systems, Inc. and its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0


import numpy as np
import pandas as pd
import torch

__all__ = ['TorchRaiDB']

# TODO: Is this still being used, keep/remove?


class TorchRaiDB(torch.utils.data.Dataset):
    def __init__(self, X, y=None, meta=None, transform=None):
        """
        Args:
            X (dataframe): pandas dataframe
            y : optional target column
        """

        if isinstance(X, pd.DataFrame):
            self.X = X.to_numpy()
        else:
            self.X = X
        self.y = y
        self.meta = meta

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):

        if not self.meta:
            if self.y:
                return self.X[idx, :], self.y[idx]
            else:
                return self.X[idx, :]
        X = []

        def onehot(x, n):
            res = [0] * n
            res[int(x)] = 1
            return res

        for i, f in enumerate(self.meta.features):
            if f.categorical:
                X.extend(onehot(self.X[idx, i], len(f.values)))
            else:
                X.append(self.X[idx, i])

        if self.y is not None:
            return np.array(X, dtype="float64"), self.y[idx].astype("float64")
        else:
            return np.array(X, dtype="float64")
