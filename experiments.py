import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
from matplotlib import rc
from pathos.multiprocessing import ProcessingPool as Pool

from temperature_scaling import TemperatureScaling, ETScaling
from errors import BS, sECE, sTCE, sCWCE, sKDECE, sKSCE, dEMsTCE, MMCE, SKCE

import os
# supposedly makes multithreading quicker, true though?
os.environ["OPENBLAS_MAIN_FREE"] = "1"
pd.options.mode.chained_assignment = None

# all errors are squared; we will take the root during plotting
errors = {
    'BS': BS,
    'KDE CE': sKDECE,
    'KS CE': sKSCE,
    'd EM 15b TCE': dEMsTCE,
    'ECE': sECE,
    '100b TCE': lambda l, y: sTCE(l, y, bins=100),
    '15b CWCE': sCWCE,
    '100b CWCE': lambda l, y: sCWCE(l, y, bins=100),
    # 'MMCE': MMCE,
    # 'SKCE': SKCE,
}


class Experiments:

    def __init__(self):
        # never have 2 different exp_id with the same exp_seed
        self.columns = [
            'setting', 'ce_value', 'ce_RC_value', 'ce_type', 'test_size',
            'runtime', 'exp_seed', 'exp_id']
        self.results_df = pd.DataFrame(columns=self.columns)
        # colors for the plots
        self.palette = {
            'KDE TCE$_2$': '#80583E',
            'RBS (ours)': '#e63946', 'KS': '#949400',
            '15b d TCE$_2$': '#A86FC9', 'ECE': '#FF8800',
            '100b TCE$_2$': '#8CE3AA', '15b CWCE$_2$': '#1A92CA',
            '100b CWCE$_2$': '#00335A', 'MMCE': '#000000'}
        rest_size = 3
        # line sizes for the plots
        self.sizes = {
            'upper bound (ours)': 5, 'RBS (ours)': 5,
            'KDE TCE$_2$': rest_size, 'KS': rest_size,
            '15b d TCE$_2$': rest_size, 'ECE': rest_size,
            '100b TCE$_2$': rest_size, '15b CWCE$_2$': rest_size,
            '100b CWCE$_2$': rest_size, 'MMCE': rest_size}

    def save(self, filename):
        self.results_df.to_csv(filename, index=False)

    def load(self, filename, append=True):
        loaded_df = pd.read_csv(filename)
        if append:
            self.results_df = self.results_df.append(loaded_df)
        else:
            self.results_df = loaded_df

    def reset(self):
        self.results_df = pd.DataFrame(columns=self.columns)

    def _eval_run(
        self, ce_type, ce_func, lgts, y, lgts_RC, s, setting, exp_id, seed,
        method
    ):
        start_t = time.time()
        v = ce_func(lgts, y)
        v_RC = ce_func(lgts_RC, y)
        t = (time.time() - start_t) / 2

        return {
            'RC_method': method,
            'setting': setting,
            'ce_value': v,
            'ce_RC_value': v_RC,
            'ce_type': ce_type,
            'test_size': s,
            'runtime': t,
            'exp_seed': seed,
            'exp_id': exp_id
        }

    def add_experiment(
        self, ds, setting, method, start_n=1000, start_rep=2500, seed=None,
        ce_types=None
    ):
        # use start time as ID
        exp_id = time.time()

        if seed is None:
            seed = int(exp_id)
        if ce_types is None:
            ce_types = list(errors.keys())

        np.random.seed(seed)
        n_ticks = 10

        logits_val, labels_val = ds[0]
        logits_test, labels_test = ds[1]
        n = logits_test.shape[0]

        if method == 'TS':
            RC_model = TemperatureScaling()
        elif method == 'ETS':
            RC_model = ETScaling()
        else:
            raise NotImplementedError()

        RC_model.fit(logits_val, labels_val)
        logits_test_RC = RC_model.predict(logits_test)

        sizes = np.rint(np.flip(np.logspace(np.log2(start_n), np.log2(n), n_ticks, base=2))).astype(int)
        # quadratically decrease repetitions
        repetitions = np.rint(np.linspace(1, np.sqrt(start_rep), n_ticks) ** 2).astype(int)

        repeated_sizes = [s for s, r in zip(sizes, repetitions) for _ in range(r)]

        def iter_func(s):
            indices = np.random.choice(n, size=s, replace=False)
            ls = logits_test[indices]
            ys = labels_test[indices]
            ls_RC = logits_test_RC[indices]

            runs = []

            def add_run(ce_type, func):
                try:
                    error = self._eval_run(
                        ce_type, func, ls, ys, ls_RC, s, setting, exp_id, seed,
                        method)
                    runs.append(error)
                except AssertionError:
                    pass

            for ce_type in ce_types:
                add_run(ce_type, errors[ce_type])

            return pd.DataFrame(runs)

        # pool = Pool()
        # results = pool.map(iter_func, repeated_sizes)
        # results = map(iter_func, repeated_sizes)
        results = []
        import pdb; pdb.set_trace()
        for rs in repeated_sizes:
            results.append(iter_func(rs))
        results = pd.concat(results)
        self.results_df = self.results_df.append(results)

    def add_DIAG_experiment(
        self, logits, labels, logits_RC, setting, start_n=1000, start_rep=2500,
        seed=None, ce_types=None
    ):
        # use start time as ID
        exp_id = time.time()

        if seed is None:
            seed = int(exp_id)
        if ce_types is None:
            ce_types = list(errors.keys())

        np.random.seed(seed)
        n_ticks = 10

        n = logits.shape[0]

        sizes = np.rint(np.flip(np.logspace(np.log2(start_n), np.log2(n), n_ticks, base=2))).astype(int)
        # quadratically decrease repetitions
        repetitions = np.rint(np.linspace(1, np.sqrt(start_rep), n_ticks) ** 2).astype(int)

        repeated_sizes = [s for s, r in zip(sizes, repetitions) for _ in range(r)]

        def iter_func(s):
            indices = np.random.choice(n, size=s, replace=False)
            ls = logits[indices]
            ys = labels[indices]
            ls_RC = logits_RC[indices]

            runs = []

            def add_run(ce_type, func):
                try:
                    error = self._eval_run(
                        ce_type, func, ls, ys, ls_RC, s, setting, exp_id, seed,
                        method='DIAG')
                    runs.append(error)
                except AssertionError:
                    pass

            for ce_type in ce_types:
                add_run(ce_type, errors[ce_type])

            return pd.DataFrame(runs)

        # pool = Pool()
        # results = pool.map(iter_func, repeated_sizes)
        # results = map(iter_func, repeated_sizes)
        results = []
        for rs in repeated_sizes:
            results.append(iter_func(rs))
        results = pd.concat(results)
        self.results_df = self.results_df.append(results)

    def saving_plot(self, plot, save_file, tight=True):
        if save_file is not None:
            if tight:
                plot.get_figure().savefig(
                    'plots/{}.png'.format(save_file), bbox_inches='tight')
                plt.close()
            else:
                plot.get_figure().savefig(
                    'plots/{}.png'.format(save_file))
                plt.close()

    def get_legend(
        self, size=(2.5, 2.5), ce_types=None, save_file=None, font_scale=1.3,
        padding=7
    ):

        rc = {'figure.figsize': size}
        sns.set(rc=rc, font_scale=font_scale, context='paper')
        sns.set_style("white")

        if ce_types is None:
            df = self.results_df[
                self.results_df['setting'] == 'densenet40_c10']
        else:
            df = self.results_df[
                (self.results_df['setting'] == 'densenet40_c10')
                & (self.results_df['ce_type'].isin(ce_types))
            ]

        df['ce_type'] = df['ce_type'].replace({
            'BS': 'RBS (ours)', 'KDE CE': 'KDE TCE$_2$',
            'KS CE': 'KS',
            'd EM 15b TCE': '15b d TCE$_2$', '100b TCE': '100b TCE$_2$',
            '15b CWCE': '15b CWCE$_2$', '100b CWCE': '100b CWCE$_2$'})
        df = df.reset_index(drop=True)

        plot = sns.lineplot(
            data=df, x='test_size', y='ce_value', hue='ce_type',
            palette=self.palette,
            sizes=self.sizes,
            size='ce_type',
            style='ce_type',
            markers=True)
        # hide the plot behind the legend
        plot.set(
            xticklabels=[], xlabel=None, ylabel=None, ylim=(-100, 100),
            xlim=(-100000, 100000))
        plot.set_yticks([])
        plot.legend(
            loc='center', framealpha=1, frameon=True, fancybox=True,
            borderpad=padding, title='Calibration Estimator')
        sns.despine(left=True, bottom=True)
        self.saving_plot(plot, save_file, tight=False)

    def plot_CE(
        self, setting, size=(4.5, 3), use_root=True, ce_types=None,
        save_file=None, legend=False, font_scale=1.4
    ):
        """
        save_file: Str, no parent folders and suffix (file type)
        """
        p = .5 if use_root else 1
        rc = {'figure.figsize': size}
        sns.set(rc=rc, font_scale=font_scale, context='paper')
        sns.set_style("white")

        if ce_types is None:
            df = self.results_df[self.results_df['setting'] == setting]
        else:
            df = self.results_df[
                (self.results_df['setting'] == setting)
                & (self.results_df['ce_type'].isin(ce_types))
            ]

        df['ce_type'] = df['ce_type'].replace({
            'BS': 'RBS (ours)', 'KDE CE': 'KDE TCE$_2$',
            'KS CE': 'KS',
            'd EM 15b TCE': '15b d TCE$_2$', '100b TCE': '100b TCE$_2$',
            '15b CWCE': '15b CWCE$_2$', '100b CWCE': '100b CWCE$_2$'})
        df['root_ce_value'] = df['ce_value'] ** p
        df = df.reset_index(drop=True)

        plot = sns.lineplot(
            data=df, x='test_size', y='root_ce_value', hue='ce_type',
            palette=self.palette,
            sizes=self.sizes,
            size='ce_type',
            style='ce_type',
            markers=True)
        ticks = df['test_size'].unique()
        plot.set(
            xscale='log', xticks=ticks.astype(float), xticklabels=ticks,
            ylim=(0, None), xlabel='Test set size', ylabel='Cal. Error')
        plot.grid(axis='y')
        if not legend:
            plot.legend([], [], frameon=False)
        else:
            plot.legend(title='')
        plot.tick_params(axis='x', rotation=45)
        sns.despine()
        self.saving_plot(plot, save_file)

    def plot_RC_delta(
        self, setting, size=(4.5, 3), use_root=True, ce_types=None,
        save_file=None, legend=False, font_scale=1.4
    ):
        """
        save_file: Str, no parent folders and suffix (file type)
        """
        p = .5 if use_root else 1
        rc = {'figure.figsize': size}
        sns.set(rc=rc, font_scale=font_scale, context='paper')
        sns.set_style("white")

        if ce_types is None:
            df = self.results_df[self.results_df['setting'] == setting]
        else:
            df = self.results_df[
                (self.results_df['setting'] == setting)
                & (self.results_df['ce_type'].isin(ce_types))
            ]
        df['ce_type'] = df['ce_type'].replace({
            'BS': 'RBS (ours)', 'KDE CE': 'KDE TCE$_2$',
            'KS CE': 'KS',
            'd EM 15b TCE': '15b d TCE$_2$', '100b TCE': '100b TCE$_2$',
            '15b CWCE': '15b CWCE$_2$', '100b CWCE': '100b CWCE$_2$'})
        # Cal. Improvement of TS in the root CE space
        df['RC_delta'] = df['ce_value'] ** p - df['ce_RC_value'] ** p
        plot = sns.lineplot(
            data=df, x='test_size', y='RC_delta', hue='ce_type',
            palette=self.palette,
            sizes=self.sizes,
            size='ce_type',
            style='ce_type',
            markers=True)
        ticks = df['test_size'].unique()
        plot.set(
            xscale='log', xticks=ticks.astype(float), xticklabels=ticks,
            xlabel='Test set size', ylabel='Cal. Improvement')
        if not legend:
            plot.legend([], [], frameon=False)
        else:
            plot.legend(title='')
        plot.grid(axis='y')
        plot.tick_params(axis='x', rotation=45)
        sns.despine()
        self.saving_plot(plot, save_file)

    def lineplot_RC_sbias(
        self, settings, size=(9.2, 7.2), use_root=True, ce_types=None,
        save_file=None
    ):
        """
        self bias
        save_file: Str, no parent folders and suffix (file type)
        """
        p = .5 if use_root else 1
        sns.set(rc={'figure.figsize': size, 'text.usetex': True})
        sns.set_style("white")

        if ce_types is None:
            ce_types = self.results_df['ce_type'].unique()
        if settings is None:
            settings = self.results_df['setting'].unique()

        df = self.results_df[
            (self.results_df['ce_type'].isin(ce_types))
            & (self.results_df['setting'].isin(settings))
        ]

        df['ce_type'] = df['ce_type'].replace({
            'BS': 'upper bound (ours)', 'KDE CE': 'KDE TCE', 'KS CE': 'KS'})
        # Cal. Improvement of TS in the root CE space
        df['RC_delta'] = df['ce_value'] ** p - df['ce_RC_value'] ** p
        keep = ['test_size', 'RC_method', 'setting', 'ce_type']
        df = df.groupby(keep)['RC_delta'].mean().reset_index()
        best_deltas = df[
            df['test_size'].isin([2897, 10000, 25000, 26032])]

        # god have mercy with my cpu
        for RC_method in df['RC_method'].unique():
            valid_settings = df[
                df['RC_method'] == RC_method]['setting'].unique()
            for setting in valid_settings:
                for error in df['ce_type'].unique():
                    rows_delta = (best_deltas['setting'] == setting) & \
                        (best_deltas['RC_method'] == RC_method) & \
                        (best_deltas['ce_type'] == error)
                    best_delta = best_deltas.loc[rows_delta, 'RC_delta'].iloc[0]
                    rows_df = (df['setting'] == setting) & \
                        (df['RC_method'] == RC_method) & \
                        (df['ce_type'] == error)
                    df.loc[rows_df, 'RC_self_bias'] = (
                        df.loc[df['setting'] == setting, 'RC_delta'] - best_delta)
        plot = sns.lineplot(
            data=df, x='test_size', y='RC_self_bias', hue='ce_type',
            palette=self.palette,
            sizes=self.sizes,
            size='ce_type',
            style='ce_type',
            markers=True)
        plot.set(xscale='log')
        ticks = df['test_size'].unique()
        plot.set(xticks=ticks.astype(float))
        plot.set(xticklabels=ticks)
        plot.grid(axis='y')
        sns.despine()
        self.saving_plot(plot, save_file)

    def lineplot_RC_rsbias(
        self, settings, size=(9.2, 7.2), use_root=True, ce_types=None,
        save_file=None
    ):
        """
        relative self bias
        save_file: Str, no parent folders and suffix (file type)
        """
        p = .5 if use_root else 1
        sns.set(rc={'figure.figsize': size, 'text.usetex': True})
        sns.set_style("white")

        if ce_types is None:
            ce_types = self.results_df['ce_type'].unique()
        if settings is None:
            settings = self.results_df['setting'].unique()

        df = self.results_df[
            (self.results_df['ce_type'].isin(ce_types))
            & (self.results_df['setting'].isin(settings))
        ]

        df['ce_type'] = df['ce_type'].replace({
            'BS': 'upper bound (ours)', 'KDE CE': 'KDE TCE', 'KS CE': 'KS'})
        # Cal. Improvement of TS in the root CE space
        df['RC_delta'] = df['ce_value'] ** p / df['ce_RC_value'] ** p
        keep = ['test_size', 'RC_method', 'setting', 'ce_type']
        df = df.groupby(keep)['RC_delta'].mean().reset_index()
        best_deltas = df[
            df['test_size'].isin([2897, 10000, 25000, 26032])]

        # god have mercy with my cpu
        for RC_method in df['RC_method'].unique():
            valid_settings = df[
                df['RC_method'] == RC_method]['setting'].unique()
            for setting in valid_settings:
                for error in df['ce_type'].unique():
                    rows_delta = (best_deltas['setting'] == setting) & \
                        (best_deltas['RC_method'] == RC_method) & \
                        (best_deltas['ce_type'] == error)
                    best_delta = best_deltas.loc[rows_delta, 'RC_delta'].iloc[0]
                    rows_df = (df['setting'] == setting) & \
                        (df['RC_method'] == RC_method) & \
                        (df['ce_type'] == error)
                    df.loc[rows_df, 'RC_bias'] = (
                        df.loc[df['setting'] == setting, 'RC_delta'] - best_delta)

        plot = sns.lineplot(
            data=df, x='test_size', y='RC_bias', hue='ce_type',
            palette=self.palette,
            sizes=self.sizes,
            size='ce_type',
            style='ce_type',
            markers=True)
        plot.set(xscale='log')
        ticks = df['test_size'].unique()
        plot.set(xticks=ticks.astype(float))
        plot.set(xticklabels=ticks)
        plot.grid(axis='y')
        sns.despine()
        self.saving_plot(plot, save_file)

    def lineplot_rsbias(
        self, settings, size=(9.2, 7.2), use_root=True, ce_types=None,
        save_file=None, legend=False, font_scale=1.15
    ):
        """
        relative self bias
        save_file: Str, no parent folders and suffix (file type)
        """
        p = .5 if use_root else 1
        rc = {'figure.figsize': size}
        sns.set(rc=rc, font_scale=font_scale, context='paper')
        sns.set_style("white")

        if ce_types is None:
            ce_types = self.results_df['ce_type'].unique()
        if settings is None:
            settings = self.results_df['setting'].unique()

        df = self.results_df[
            (self.results_df['ce_type'].isin(ce_types))
            & (self.results_df['setting'].isin(settings))
        ]

        df['ce_type'] = df['ce_type'].replace({
            'BS': 'RBS (ours)', 'KDE CE': 'KDE TCE$_2$',
            'KS CE': 'KS',
            'd EM 15b TCE': '15b d TCE$_2$', '100b TCE': '100b TCE$_2$',
            '15b CWCE': '15b CWCE$_2$', '100b CWCE': '100b CWCE$_2$'})
        # Cal. Improvement of TS in the root CE space
        df['RC_delta'] = df['ce_value'] ** p
        keep = ['test_size', 'RC_method', 'setting', 'ce_type']
        df = df.groupby(keep)['RC_delta'].mean().reset_index()
        best_deltas = df[
            df['test_size'].isin([2897, 10000, 25000, 26032])]

        # god have mercy with my cpu
        for RC_method in df['RC_method'].unique():
            valid_settings = df[
                df['RC_method'] == RC_method]['setting'].unique()
            for setting in valid_settings:
                for error in df['ce_type'].unique():
                    rows_delta = (best_deltas['setting'] == setting) & \
                        (best_deltas['RC_method'] == RC_method) & \
                        (best_deltas['ce_type'] == error)
                    best_delta = best_deltas.loc[rows_delta, 'RC_delta'].iloc[0]
                    rows_df = (df['setting'] == setting) & \
                        (df['RC_method'] == RC_method) & \
                        (df['ce_type'] == error)
                    df.loc[rows_df, 'relative_self_bias'] = (
                        df.loc[df['setting'] == setting, 'RC_delta'] / best_delta)

        plot = sns.lineplot(
            data=df, x='test_size', y='relative_self_bias', hue='ce_type',
            palette=self.palette,
            sizes=self.sizes,
            size='ce_type',
            style='ce_type',
            markers=True)
        if not legend:
            plot.legend([], [], frameon=False)
        else:
            plot.legend(title='')
        ticks = df['test_size'].unique()
        plot.set(
            xscale='log', xticks=ticks.astype(float), xticklabels=ticks,
            xlabel='Test set size', ylabel='Relative bias')
        plot.grid(axis='y')
        plot.tick_params(axis='x', rotation=45)
        sns.despine()
        self.saving_plot(plot, save_file)

    def lineplot_RC_bias(
        self, settings, size=(9.2, 7.2), use_root=False, ce_types=None,
        save_file=None
    ):
        """
        save_file: Str, no parent folders or suffix (file type)
        """
        # use the square root is actually false here as this is not the bias
        # anymore
        p = 1 # .5 if use_root else 1
        sns.set(rc={'figure.figsize': size, 'text.usetex': True})
        sns.set_style("white")

        if ce_types is None:
            ce_types = self.results_df['ce_type'].unique()
        if settings is None:
            settings = self.results_df['setting'].unique()

        df = self.results_df[
            (self.results_df['ce_type'].isin(ce_types))
            & (self.results_df['setting'].isin(settings))
        ]

        # Cal. Improvement of RC in the squared space
        df['RC_delta'] = df['ce_value'] ** p - df['ce_RC_value'] ** p
        keep = ['test_size', 'RC_method', 'setting', 'ce_type']
        df = df.groupby(keep)['RC_delta'].mean().reset_index()
        best_deltas = df[
            (df['test_size'].isin([2897, 10000, 25000, 26032]))
            & (df['ce_type'] == 'BS')]

        for RC_method in df['RC_method'].unique():
            valid_settings = df[
                df['RC_method'] == RC_method]['setting'].unique()
            for setting in valid_settings:
                rows_delta = (best_deltas['setting'] == setting) & (best_deltas['RC_method'] == RC_method)
                best_delta = best_deltas.loc[rows_delta, 'RC_delta'].iloc[0]
                rows_df = (df['setting'] == setting) & (df['RC_method'] == RC_method)
                df.loc[rows_df, 'RC_bias'] = (
                    df.loc[df['setting'] == setting, 'RC_delta'] - best_delta)

        df['ce_type'] = df['ce_type'].replace({
            'BS': 'upper bound (ours)', 'KDE CE': 'KDE TCE', 'KS CE': 'KS'})
        plot = sns.lineplot(
            data=df, x='test_size', y='RC_bias', hue='ce_type',
            palette=self.palette,
            sizes=self.sizes,
            size='ce_type',
            style='ce_type',
            markers=True)
        plot.set(xscale='log')
        ticks = df['test_size'].unique()
        plot.set(xticks=ticks.astype(float))
        plot.set(xticklabels=ticks)
        plot.grid(axis='y')
        sns.despine()
        self.saving_plot(plot, save_file)

    def plot_runtime(
        self, setting, size=(12.8, 9.6), ce_types=None, log_y=False,
        save_file=None
    ):
        """
        save_file: Str, no parent folders and suffix (file type)
        """
        sns.set(rc={'figure.figsize': size, 'text.usetex': True})

        if ce_types is None:
            df = self.results_df[self.results_df['setting'] == setting]
        else:
            df = self.results_df[
                (self.results_df['setting'] == setting)
                & (self.results_df['ce_type'].isin(ce_types))
            ]
        df['ce_type'] = df['ce_type'].replace({
            'BS': 'upper bound (ours)', 'KDE CE': 'KDE TCE', 'KS CE': 'KS'})
        plot = sns.lineplot(
            data=df, x='test_size', y='runtime', hue='ce_type')
        if log_y:
            plot.set(yscale='log')
        plot.set(xscale='log')
        ticks = df['test_size'].unique()
        plot.set(xticks=ticks.astype(float))
        plot.set(xticklabels=ticks)
        self.saving_plot(plot, save_file)

    def boxplot(
        self, settings=None, ce_types=None, size=(9.2, 7.2),
        use_root=True, save_file=None, set_size='high'
    ):

        sns.set(rc={'figure.figsize': size, 'text.usetex': True})
        sns.set_style("white")
        # not my proudest moment...
        if (ce_types is None) and (settings is None):
            df = self.results_df
        elif (ce_types is None) and (settings is not None):
            df = self.results_df[
                (self.results_df['setting'].isin(settings))
            ]
        elif ce_types is not None and settings is None:
            df = self.results_df[
                (self.results_df['ce_type'].isin(ce_types))
            ]
        else:
            df = self.results_df[
                (self.results_df['ce_type'].isin(ce_types))
                & (self.results_df['setting'].isin(settings))
            ]
        p = .5 if use_root else 1
        df['root_ce_value'] = df['ce_value'] ** p
        # this only works cause the ticks differ per dataset
        if set_size == 'max':
            df = df[df['test_size'].isin([2897, 10000, 25000, 26032])]
        elif set_size == 'high':
            df = df[df['test_size'].isin([1993, 5995, 13536, 14032])]
        elif set_size == 'min':
            df = df[df['test_size'] == 100]

        df['ce_type'] = df['ce_type'].replace({
            'BS': 'upper bound (ours)', 'KDE CE': 'KDE TCE', 'KS CE': 'KS'})
        plot = sns.boxplot(
            data=df, y='root_ce_value', x='setting', hue='ce_type')
        plot.set(ylim=(0, 0.8))
        sns.despine()
        self.saving_plot(plot, save_file)

    def boxplot_RC_delta(
        self, settings=None, ce_types=None, size=(9.2, 7.2),
        use_root=True, save_file=None, set_size='high'
    ):
        """
        save_file: Str, no parent folders and suffix (file type)
        """

        sns.set(rc={'figure.figsize': size, 'text.usetex': True})
        sns.set_style("white")
        if ce_types is None:
            ce_types = self.results_df['ce_type'].unique()
        if settings is None:
            settings = self.results_df['setting'].unique()

        df = self.results_df[
            (self.results_df['ce_type'].isin(ce_types))
            & (self.results_df['setting'].isin(settings))
        ]

        p = .5 if use_root else 1
        # Cal. Improvement of TS in the root CE space
        df['RC_delta'] = df['ce_value'] ** p - df['ce_RC_value'] ** p
        # this only works cause the ticks differ per dataset
        if set_size == 'max':
            df = df[df['test_size'].isin([2897, 10000, 25000, 26032])]
        elif set_size == 'high':
            df = df[df['test_size'].isin([1993, 5995, 13536, 14032])]
        elif set_size == 'min':
            df = df[df['test_size'] == 100]

        df['ce_type'] = df['ce_type'].replace({
            'BS': 'upper bound (ours)', 'KDE CE': 'KDE TCE', 'KS CE': 'KS'})
        plot = sns.boxplot(
            data=df, y='RC_delta', x='setting', hue='ce_type')
        sns.despine()
        self.saving_plot(plot, save_file)

    def boxplot_delta(
        self, settings=None, ce_types=None, size=(9.2, 7.2),
        use_root=True, save_file=None
    ):
        """
        save_file: Str, no parent folders and suffix (file type)
        """

        sns.set(rc={'figure.figsize': size, 'text.usetex': True})
        sns.set_style("white")
        if ce_types is None:
            ce_types = self.results_df['ce_type'].unique()
        if settings is None:
            settings = self.results_df['setting'].unique()

        df = self.results_df[
            (self.results_df['ce_type'].isin(ce_types))
            & (self.results_df['setting'].isin(settings))
        ]
        p = .5 if use_root else 1
        df['root_ce_value'] = df['ce_value'] ** p
        df['size_delta'] = df['ce_value'] ** p
        # this only works cause the ticks differ per dataset
        for set in df['setting'].unique():
            for typ in df['ce_type'].unique():
                # value to substract
                val = df[
                    (df['setting'] == set)
                    & (df['ce_type'] == typ)
                    & (df['test_size'].isin([2897, 10000, 25000, 26032]))
                    ]['root_ce_value'].iloc[0]
                # column to substract from
                col = df[
                    (df['setting'] == set)
                    & (df['test_size'] == 100)
                    & (df['ce_type'] == typ)
                    ]['root_ce_value']
                df.loc[
                    (df['setting'] == set)
                    & (df['ce_type'] == typ)
                    & (df['test_size'] == 100),
                    'size_delta'] = col - val
        df = df[df['test_size'] == 100]

        df['ce_type'] = df['ce_type'].replace({
            'BS': 'upper bound (ours)', 'KDE CE': 'KDE TCE', 'KS CE': 'KS'})
        plot = sns.boxplot(
            data=df, y='size_delta', x='setting', hue='ce_type')
        sns.despine()
        self.saving_plot(plot, save_file)

    def boxplot_delta_RC_delta(
        self, settings=None, ce_types=None, size=(9.2, 7.2),
        use_root=True, save_file=None
    ):
        """
        save_file: Str, no parent folders and suffix (file type)
        """

        sns.set(rc={'figure.figsize': size})
        sns.set_style("white")
        # not my proudest moment...
        if ce_types is None:
            ce_types = self.results_df['ce_type'].unique()
        if settings is None:
            settings = self.results_df['setting'].unique()

        df = self.results_df[
            (self.results_df['ce_type'].isin(ce_types))
            & (self.results_df['setting'].isin(settings))
        ]
        p = .5 if use_root else 1
        df['RC_delta'] = df['ce_value'] ** p - df['ce_RC_value'] ** p
        df['size_delta'] = df['RC_delta']
        # this only works cause the ticks differ per dataset
        for set in df['setting'].unique():
            for typ in df['ce_type'].unique():
                # value to substract
                val = df[
                    (df['setting'] == set)
                    & (df['ce_type'] == typ)
                    & (df['test_size'].isin([2897, 10000, 25000, 26032]))
                    ]['RC_delta'].iloc[0]
                # column to substract from
                col = df[
                    (df['setting'] == set)
                    & (df['test_size'] == 100)
                    & (df['ce_type'] == typ)
                    ]['RC_delta']
                df.loc[
                    (df['setting'] == set)
                    & (df['ce_type'] == typ)
                    & (df['test_size'] == 100),
                    'size_delta'] = col - val
        df = df[df['test_size'] == 100]

        df['ce_type'] = df['ce_type'].replace({
            'BS': 'upper bound (ours)', 'KDE CE': 'KDE TCE', 'KS CE': 'KS'})
        plot = sns.boxplot(
            data=df, y='size_delta', x='setting', hue='ce_type')
        sns.despine()
        self.saving_plot(plot, save_file)

    def print_results_size(self, max_rows=999):
        pd.options.display.max_rows = max_rows
        print(exp.results_df.groupby(['setting', 'ce_type']).size())
        pd.options.display.max_rows = 15


if __name__ == '__main__':

    from utils import unpickle_probs
    import argparse
    import os


    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--logits_path',
        type=str,
        required=True
    )
    parser.add_argument(
        '--setting',
        type=str,
        required=True,
        help='String identifier of dataset+model'
    )
    parser.add_argument(
        '--save_file',
        type=str,
        default='results.csv',
        help='File to append the results into; can pre-exist'
    )
    parser.add_argument(
        '--method',
        type=str,
        default='TS',
        help='Recalibration method;',
        choices=['TS', 'ETS', 'DIAG']
    )
    parser.add_argument(
        '--start_rep',
        type=int,
        default=2500,
        help='Repetitions of lowest sample size; must be quadratic number;'
    )
    parser.add_argument(
        '--ce_types',
        type=str,
        nargs="+",
        default=None,
        choices=errors.keys()
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )

    args = parser.parse_args()

    exp = Experiments()
    if os.path.isfile(args.save_file):
        exp.load(args.save_file)

    t = time.time()

    if args.method == 'DIAG':
        logits = np.load(args.logits_path + '/nll_logits.npy')
        labels = np.load(args.logits_path + '/nll_labels.npy')
        logits_RC = np.load(args.logits_path + '/nll_scores.npy')
        exp.add_DIAG_experiment(
            logits=logits, labels=labels, logits_RC=logits_RC,
            setting=args.setting, start_rep=args.start_rep,
            ce_types=args.ce_types, seed=args.seed)

    else:
        ds = unpickle_probs(args.logits_path)
        exp.add_experiment(
            ds=ds, setting=args.setting, method=args.method,
            start_rep=args.start_rep,
            ce_types=args.ce_types, seed=args.seed)

    print('Runtime [s]:', time.time() - t)
    exp.save(args.save_file)
    print(exp.results_df)
