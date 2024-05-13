# !/usr/bin/env python
# Created by "Thieu" at 08:58, 16/03/2020 ----------%
#       Email: nguyenthieu2102@gmail.com            %
#       Github: https://github.com/thieu1995        %
# --------------------------------------------------%

import numpy as np
import math
from math import gamma
from copy import deepcopy
from mealpy.utils.history import History
from mealpy.utils.problem import Problem
from mealpy.utils.termination import Termination
from mealpy.utils.logger import Logger
from mealpy.utils.validator import Validator
import concurrent.futures as parallel
import os
import time
import random
from mealpy import elm
import pandas as pd
# import optuna
from sklearn.ensemble import RandomForestRegressor
from mealpy import elm
from xgboost import XGBRegressor
import sklearn.model_selection as ms


class Optimizer:
    """
    The base class of all algorithms. All methods in this class will be inherited

    Notes
    ~~~~~
    + The function solve() is the most important method, trained the model
    + The parallel (multithreading or multiprocessing) is used in method: create_population(), update_target_wrapper_population()
    + The general format of:
        + population = [agent_1, agent_2, ..., agent_N]
        + agent = global_best = solution = [position, target]
        + target = [fitness value, objective_list]
        + objective_list = [obj_1, obj_2, ..., obj_M]
    + Access to the:
        + position of solution/agent: solution[0] or solution[self.ID_POS] or model.solution[model.ID_POS]
        + fitness: solution[1][0] or solution[self.ID_TAR][self.ID_FIT] or model.solution[model.ID_TAR][model.ID_FIT]
        + objective values: solution[1][1] or solution[self.ID_TAR][self.ID_OBJ] or model.solution[model.ID_TAR][model.ID_OBJ]
    """

    ID_POS = 0  # Index of position/location of solution/agent
    ID_TAR = 1  # Index of target list, (includes fitness value and objectives list)

    ID_FIT = 0  # Index of target (the final fitness) in fitness
    ID_OBJ = 1  # Index of objective list in target

    PERCENTAGE_BY_REAL = 0.1

    EPSILON = 10E-10

    def __init__(self, problem, kwargs=None):
        """
        Args:
            problem: an instance of Problem class or a dictionary

        Examples:
            problem = {
                "fit_func": your objective function,
                "lb": list of value
                "ub": list of value
                "minmax": "min" or "max"
                "verbose": True or False
                "n_dims": int (Optional)
                "obj_weights": list weights corresponding to all objectives (Optional, default = [1, 1, ...1])
            }
        """
        super(Optimizer, self).__init__()
        self.RF_n_estimators = 100  # int
        self.RF_criterion = "squared_error"  # “squared_error”, “absolute_error”, “poisson”
        # self.max_depth = None  # int
        # self.min_samples_split = 2  # int or float
        # self.min_samples_leaf = 1  # int or float
        # self.min_weight_fraction_leaf = 0.0  # float
        self.RF_max_features = 0.3  # “sqrt”, “log2”, None/1.0{0.3 for more randomness}
        # self.max_leaf_nodes = None
        # self.min_impurity_decrease = 0.0
        # self.bootstrap = True
        # self.oob_score = False  # only available if bootstrap true
        # self.n_jobs = None  # None means 1 unless in a joblib.parallel_backend context. -1 means using all processors.
        # self.random_state = None  # int
        # self.verbose = 0
        # self.warm_start = False
        # self.ccp_alpha = 0.0  # non-negative
        # self.max_samples = None  # int or float
        self.XGB_max_depth = 6  # the larger the value, the greater the chance of overfit
        self.XGB_eta = 0.3  # alias learning rate; step 0.1 such as [0.1, 0.2, 0.3, 0.4, 0.5] Step size shrinkage used in update to prevents overfitting.
        self.XGB_objective = 'reg:squarederror'  # reg:squarederror, reg:squaredlogerror, reg:logistic, reg:pseudohubererror, reg:absoluteerror
        # self.booster = 'gbtree'  # default gbtree; [gbtree, gblinear, dart]
        # performance metric
        self.mse_history = None


        self.ELM_C = 1
        self.ELM_hidden_units = 200  # ELM param
        self.ELM_activation_function = 'relu'  # ELM param
        self.ELM_random_type = 'normal'  # ELM param
        self.ELM_elm_type = 'reg'

        self.epoch, self.pop_size, self.solution = None, None, None
        self.mode, self.n_workers, self._print_model = None, None, ""
        self.pop, self.g_best = None, None
        if kwargs is None: kwargs = {}
        self.__set_keyword_arguments(kwargs)
        self.problem = Problem(problem=problem)
        self.amend_position = self.problem.amend_position
        self.generate_position = self.problem.generate_position
        self.logger = Logger(self.problem.log_to, log_file=self.problem.log_file).create_logger(name=f"{self.__module__}.{self.__class__.__name__}")
        self.logger.info(self.problem.msg)
        self.history = History(log_to=self.problem.log_to, log_file=self.problem.log_file)
        self.validator = Validator(log_to=self.problem.log_to, log_file=self.problem.log_file)
        if "name" in kwargs: self._print_model += f"Model: {kwargs['name']}, "
        if "fit_name" in kwargs: self._print_model += f"Func: {kwargs['fit_name']}, "
        self.termination_flag = False
        if "termination" in kwargs:
            self.termination = Termination(termination=kwargs["termination"], log_to=self.problem.log_to, log_file=self.problem.log_file)
            self.termination_flag = True
        self.nfe_per_epoch = self.pop_size
        self.sort_flag, self.count_terminate = False, None
        self.AVAILABLE_MODES = ["process", "thread", "swarm"]

    def __set_keyword_arguments(self, kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def termination_start(self):
        if self.termination_flag:
            if self.termination.mode == 'TB':
                self.count_terminate = time.perf_counter()
            elif self.termination.mode == 'ES':
                self.count_terminate = 0
            elif self.termination.mode == 'MG':
                self.count_terminate = self.epoch
            else:  # number of function evaluation (NFE)
                self.count_terminate = 0 # self.pop_size  # First out of loop
            self.logger.warning(f"Stopping condition mode: {self.termination.name}, with maximum value is: {self.termination.quantity}")

    def initialization(self, starting_positions=None):
        if starting_positions is None:
            self.pop = self.create_population(self.pop_size)
        else:
            if type(starting_positions) in [list, np.ndarray] and len(starting_positions) == self.pop_size:
                if isinstance(starting_positions[0], np.ndarray) and len(starting_positions[0]) == self.problem.n_dims:
                    self.pop = [self.create_solution(self.problem.lb, self.problem.ub, pos) for pos in starting_positions]
                else:
                    self.logger.error("Starting positions should be a list of positions or 2D matrix of positions only.")
                    exit(0)
            else:
                self.logger.error("Starting positions should be a list/2D matrix of positions with same length as pop_size hyper-parameter.")
                exit(0)

    def after_initialization(self):
        # The initial population is sorted or not depended on algorithm's strategy
        pop_temp, self.g_best = self.get_global_best_solution(self.pop)
        if self.sort_flag: self.pop = pop_temp

    def get_target_wrapper(self, position):
        """
        Args:
            position (nd.array): position (nd.array): 1-D numpy array

        Returns:
            [fitness, [obj1, obj2,...]]
        """
        if self.problem.multi_args:
            objs = self.problem.fit_func(position, self.problem.data)
        else:
            objs = self.problem.fit_func(position)
        if not self.problem.obj_is_list:
            objs = [objs]
        fit = np.dot(objs, self.problem.obj_weights)
        return [fit, objs]

    def get_target_wrapper_2(self, position):
        """
        Args:
            position (nd.array): position (nd.array): 1-D numpy array

        Returns:
            [fitness, [obj1, obj2,...]]
        """
        prob = random.uniform(0, 1)
        if self.problem.multi_args:
            if prob >= self.PERCENTAGE_FOR_REAL:
                objs = self.problem.fit_func(position, self.problem.data)
            else:
                objs = self.Cal_ELM()
        else:
            if prob >= self.PERCENTAGE_FOR_REAL:
                objs = self.problem.fit_func(position)
            else:
                objs = self.Cal_ELM()
        if not self.problem.obj_is_list:
            objs = [objs]
        fit = np.dot(objs, self.problem.obj_weights)
        return [fit, objs], prob

    def create_solution(self, lb=None, ub=None, pos=None):
        """
        To get the position, target wrapper [fitness and obj list]
            + A[self.ID_POS]                  --> Return: position
            + A[self.ID_TAR]                  --> Return: [fitness, [obj1, obj2, ...]]
            + A[self.ID_TAR][self.ID_FIT]     --> Return: fitness
            + A[self.ID_TAR][self.ID_OBJ]     --> Return: [obj1, obj2, ...]

        Args:
            lb: list of lower bound values
            ub: list of upper bound values
            pos (np.ndarray): the known position. If None is passed, the default function generate_position() will be used

        Returns:
            list: wrapper of solution with format [position, [fitness, [obj1, obj2, ...]]]
        """
        if pos is None:
            pos = self.generate_position(lb, ub)
        position = self.amend_position(pos, lb, ub)
        target = self.get_target_wrapper(position)
        return [position, target]

    def before_evolve(self, epoch):
        pass

    def evolve(self, epoch):
        pass

    def after_evolve(self, epoch):
        pass

    def termination_end(self, epoch):
        finished = False
        if self.termination_flag:
            if self.termination.mode == 'TB':
                if time.perf_counter() - self.count_terminate >= self.termination.quantity:
                    self.logger.warning(f"Stopping criterion with mode {self.termination.name} occurred. End program!")
                    finished = True
            elif self.termination.mode == 'FE':
                self.count_terminate += self.nfe_per_epoch
                if self.count_terminate >= self.termination.quantity:
                    self.logger.warning(f"Stopping criterion with mode {self.termination.name} occurred. End program!")
                    finished = True
            elif self.termination.mode == 'MG':
                if epoch >= self.termination.quantity:
                    self.logger.warning(f"Stopping criterion with mode {self.termination.name} occurred. End program!")
                    finished = True
            else:  # Early Stopping
                temp = self.count_terminate + self.history.get_global_repeated_times(self.ID_TAR, self.ID_FIT, self.EPSILON)
                if temp >= self.termination.quantity:
                    self.logger.warning(f"Stopping criterion with mode {self.termination.name} occurred. End program!")
                    finished = True
        return finished

    def check_mode_and_workers(self, mode, n_workers):
        self.mode = mode
        if n_workers is not None:
            if self.mode == "process":
                self.n_workers = self.validator.check_int("n_workers", n_workers, [2, min(61, os.cpu_count() - 1)])
            if self.mode == "thread":
                self.n_workers = self.validator.check_int("n_workers", n_workers, [2, min(32, os.cpu_count() + 4)])

    def extract_cost_and_pop(self, pop):
        cost = []
        population = []
        for p in pop:
            cost.append(p[1][0])
            population.append(p[0])
        return np.asarray(cost), np.asarray(population)

    def evaluate_pop_real(self, pop):
        pp = [item[0] for item in pop]
        cost = []
        for ind in pp:
            target = self.get_target_wrapper(ind)
            cost.append(target[0])
        return pp, cost

    def get_array_elements(self, pop, indexes):
        a = []
        for i in indexes:
            a.append(pop[i][:])
        return a


    def calculate_accuracy(self):
        return 1

    def evaluate_pop_mix_10_per(self, best_model_name,  selected_model, pop):
        no_of_individuals_by_real = math.ceil(self.pop_size * self.PERCENTAGE_BY_REAL)

        pop_real, cost_real = self.evaluate_pop_real(pop[0:no_of_individuals_by_real])
        pop = [item[0] for item in pop]
        pop = np.array(pop)
        cost = None
        if best_model_name=="ELM":
            cost = (selected_model.predict(pop)).reshape(self.pop_size)
        else:
            cost = selected_model.predict(pop)
        cost_real = np.asarray(cost_real)

        m1 = 0
        m2 = 0
        accuracy_ = 0
        if cost_real.shape[0] >= 1:
            # accuracy
            m1 = np.mean(cost_real)
            m2 = (np.sqrt(np.sum(
                (cost_real - cost[0:no_of_individuals_by_real]) * (cost_real - cost[0:no_of_individuals_by_real])) /
                          cost_real.shape[0]))
            accuracy_ = 1 - (m2 / m1)
            cost[0:no_of_individuals_by_real] = cost_real
        self.mse_history[best_model_name].append(accuracy_)
        return cost_real, np.asarray(pop_real), cost, pop, (m1, m2), accuracy_

    def evaluate_pop_mix(self, modelElm, pop):
        probs = np.random.rand(self.pop_size)
        probs_sorted = probs <= self.PERCENTAGE_BY_REAL
        probs_sorted = probs_sorted.astype(int)
        porbs_index_real = np.where(probs_sorted == 1)[0]
        p = self.get_array_elements(pop, porbs_index_real)
        pop_real, cost_real = self.evaluate_pop_real(p)
        pop = [item[0] for item in pop]
        pop = np.array(pop)
        cost = (modelElm.predict(pop)).reshape(self.pop_size)
        cost_real = np.asarray(cost_real)

        m1 = 0
        m2 = 0
        accuracy_ = 0
        if cost_real.shape[0] >= 1:
            # accuracy
            m1 = np.mean(cost_real)
            m2 = (np.sqrt(np.sum(
                (cost_real - cost[porbs_index_real]) * (cost_real - cost[porbs_index_real])) /
                          cost_real.shape[0]))
            accuracy_ = 1 - (m2 / m1)
            cost[porbs_index_real] = cost_real

        return cost_real, np.asarray(pop_real), cost, pop, (m1, m2), accuracy_

    # update the working population based on DE
    def update_current_pop(self, cost, population):
        for i in range(self.pop_size):
            self.pop[i] = self.get_better_solution([population[i], [cost[i], [cost[i]]]], self.pop[i])

    def solve(self, mode='single', starting_positions=None, n_workers=None):
        """
        Args:
            mode (str): Parallel: 'process', 'thread'; Sequential: 'swarm', 'single'.

                * 'process': The parallel mode with multiple cores run the tasks
                * 'thread': The parallel mode with multiple threads run the tasks
                * 'swarm': The sequential mode that no effect on updating phase of other agents
                * 'single': The sequential mode that effect on updating phase of other agents, default

            starting_positions(list, np.ndarray): List or 2D matrix (numpy array) of starting positions with length equal pop_size hyper_parameter
            n_workers (int): The number of workers (cores or threads) to do the tasks (effect only on parallel mode)

        Returns:
            list: [position, fitness value]
        """
        self.check_mode_and_workers(mode, n_workers)
        self.termination_start()
        self.initialization(starting_positions)
        self.after_initialization()
        self.history.store_initial_best(self.g_best)

        cost, population = self.extract_cost_and_pop(self.pop)
        df_pop = pd.DataFrame(population)
        df_cost = pd.DataFrame(cost)

        models = {
            "RF": RandomForestRegressor,
            "XGB": XGBRegressor,
            "ELM": elm.elm
        }
        # surrogate_models = {name: model() for name, model in models.items()}
        elm_model = models["ELM"](hidden_units=self.ELM_hidden_units,
                                     activation_function=self.ELM_activation_function,
                                     random_type=self.ELM_random_type,
                                     x=df_pop.values, y=df_cost.values,
                                     C=self.ELM_C, elm_type=self.ELM_elm_type)
        beta, train_score, running_time = elm_model.fit("no_re")
        cost_elm = (elm_model.predict(population)).reshape(self.pop_size)
        xgb_model = models["XGB"](max_depth=self.XGB_max_depth,
                                     eta=self.XGB_eta,
                                     objective=self.XGB_objective)
        xgb_model.fit(df_pop, df_cost.values.ravel())
        cost_xgb= xgb_model.predict(population)
        rf_model = models["RF"](n_estimators=self.RF_n_estimators,
                                     criterion=self.RF_criterion,
                                     max_features=self.RF_max_features)
        rf_model.fit(df_pop, df_cost.values.ravel())
        cost_rf= rf_model.predict(population)

        self.mse_history = {name: [] for name in models}
        m1 = 0
        m2 = 0
        accuracy_ = 0
        m1 = np.mean(cost)
        m2 = (np.sqrt(np.sum((cost - cost_elm) * (cost - cost_elm)) / cost.shape[0]))
        accuracy_ = 1 - (m2 / m1)
        self.mse_history["ELM"].append(accuracy_)
        m2 = (np.sqrt(np.sum((cost - cost_xgb) * (cost - cost_xgb)) / cost.shape[0]))
        accuracy_ = 1 - (m2 / m1)
        self.mse_history["XGB"].append(accuracy_)
        m2 = (np.sqrt(np.sum((cost - cost_rf) * (cost - cost_rf)) / cost.shape[0]))
        accuracy_ = 1 - (m2 / m1)
        self.mse_history["RF"].append(accuracy_)

        rmse = [("mean obj of the current iteration", "rmse")]
        accuracy_arr = []
        accuracy_arr = np.array(accuracy_arr)

        for epoch in range(0, self.epoch):
            time_epoch = time.perf_counter()
            self.before_evolve(epoch)
            population = self.evolve(epoch)
            self.after_evolve(epoch)

            if epoch <= 2:
                population, cost = self.evaluate_pop_real(population)
                df_pop_new = pd.DataFrame(population)
                df_pop = pd.concat([df_pop, df_pop_new], axis=0)
                df_cost_new = pd.DataFrame(cost)
                df_cost = pd.concat([df_cost, df_cost_new], axis=0)
            else:
                # Model selection based on historical MSE values
                best_model_name = max(self.mse_history, key=lambda x: np.mean(self.mse_history[x]))
                # print(best_model_name)
                best_model = models[best_model_name]
                # Run the best model based on its identifier
                if best_model_name == "ELM":
                    selected_model = best_model(hidden_units=self.ELM_hidden_units,
                                                activation_function=self.ELM_activation_function,
                                                random_type=self.ELM_random_type,
                                                x=df_pop.values, y=df_cost.values,
                                                C=self.ELM_C, elm_type=self.ELM_elm_type)
                    beta, train_score, running_time = selected_model.fit("no_re")
                elif best_model_name == "XGB":
                    selected_model = best_model(max_depth=self.XGB_max_depth,
                                                eta=self.XGB_eta,
                                                objective=self.XGB_objective)
                    selected_model.fit(df_pop, df_cost.values.ravel())
                elif best_model_name == "RF":
                    selected_model = best_model(n_estimators=self.RF_n_estimators,
                                                criterion=self.RF_criterion,
                                                max_features=self.RF_max_features)
                    selected_model.fit(df_pop, df_cost.values.ravel())

                #
                # if selected_model_identifier == "Elm":
                #     selected_model = model_class(hidden_units=self.ELM_hidden_units,
                #                                  activation_function=self.ELM_activation_function,
                #                                  random_type=self.ELM_random_type,
                #                                  x=df_pop.values, y=df_cost.values,
                #                                  C=self.ELM_C, elm_type=self.ELM_elm_type)
                #     beta, train_score, running_time = selected_model.fit("no_re")
                # elif selected_model_identifier == "XGB":
                #     selected_model = model_class(max_depth=self.XGB_max_depth,
                #                                  eta=self.XGB_eta,
                #                                  objective=self.XGB_objective)
                #     selected_model.fit(df_pop, df_cost.values.ravel())
                # elif selected_model_identifier == "RF":
                #     selected_model = model_class(n_estimators=self.RF_n_estimators,
                #                                  criterion=self.RF_criterion,
                #                                  max_features=self.RF_max_features)
                #     selected_model.fit(df_pop, df_cost.values.ravel())

                # print(selected_model_identifier)
                # if selected_model_identifier == "Elm":
                #     selected_model  = elm.elm(hidden_units=self.ELM_hidden_units, activation_function=self.ELM_activation_function, random_type=self.ELM_random_type, x=df_pop.values, y=df_cost.values, C=self.ELM_C, elm_type=self.ELM_elm_type)
                #     beta, train_score, running_time = selected_model.fit("no_re")
                # elif selected_model_identifier == "XGB":
                #     selected_model = XGBRegressor(max_depth=self.ELM_max_depth, eta=self.ELM_eta, objective=self.ELM_objective)
                #     selected_model.fit(df_pop, df_cost.values.ravel())
                # elif selected_model_identifier == "RF":
                #     selected_model = RandomForestRegressor(n_estimators=self.RF_n_estimators, criterion=self.RF_criterion, max_features=self.RF_max_features)
                #     selected_model.fit(df_pop, df_cost.values.ravel())

                cost_real, pop_real, cost, population, mse, accuracy_ = \
                    self.evaluate_pop_mix_10_per(best_model_name, selected_model, population)
                rmse.append(mse)
                accuracy_arr = np.append(accuracy_arr, accuracy_)

                df_cost_new = pd.DataFrame(cost_real)
                df_cost = pd.concat([df_cost, df_cost_new], axis=0)
                df_pop_new = pd.DataFrame(pop_real)
                df_pop = pd.concat([df_pop, df_pop_new.iloc[0:pop_real.shape[0]]], axis=0)

            self.update_current_pop(cost, population)
            # Update global best position, the population is sorted or not depended on algorithm's strategy
            pop_temp, self.g_best = self.update_global_best_solution(self.pop)
            if self.sort_flag: self.pop = pop_temp

            time_epoch = time.perf_counter() - time_epoch
            self.track_optimize_step(self.pop, epoch+1, time_epoch)
            if self.termination_end(epoch+1):
                break
        self.track_optimize_process()
        population, cost = self.evaluate_pop_real(self.pop)
        population = np.asarray(population)
        cost = np.asarray(cost)
        ind = np.argsort(cost)
        cost = cost[ind]
        population = population[ind, :]
        real_best_cost = cost[0]
        resl_best_position = population[0]

        return resl_best_position, real_best_cost,  np.mean(accuracy_arr) # self.solution[self.ID_POS], self.solution[self.ID_TAR][self.ID_FIT]

    def objective(self, trial, df_pop, df_cost):
        n_estimators = trial.suggest_int('n_estimators', 50, 150)
        criterion = trial.suggest_categorical('criterion',
                                                        ['squared_error', 'absolute_error', 'poisson'])
        max_features = trial.suggest_float('max_features', 0.1, 1.0)

        modelRF = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_features=max_features)
        # modelRF.fit(df_pop, df_cost)
        # modelRF.predit(df_pop)
        score = ms.cross_val_score(modelRF, df_pop, df_cost.ravel(), cv=5, scoring="r2")
        r2_mean = score.mean()
        return r2_mean


    def track_optimize_step(self, population=None, epoch=None, runtime=None):
        """
        Save some historical data and print out the detailed information of training process

        Args:
            population (list): the current population
            epoch (int): current iteration
            runtime (float): the runtime for current iteration
        """
        ## Save history data
        pop = deepcopy(population)
        if self.problem.save_population:
            self.history.list_population.append(pop)
        self.history.list_epoch_time.append(runtime)
        self.history.list_global_best_fit.append(self.history.list_global_best[-1][self.ID_TAR][self.ID_FIT])
        self.history.list_current_best_fit.append(self.history.list_current_best[-1][self.ID_TAR][self.ID_FIT])
        # Save the exploration and exploitation data for later usage
        pos_matrix = np.array([agent[self.ID_POS] for agent in pop])
        div = np.mean(np.abs(np.median(pos_matrix, axis=0) - pos_matrix), axis=0)
        self.history.list_diversity.append(np.mean(div, axis=0))
        ## Print epoch
        self.logger.info(f">{self._print_model}Epoch: {epoch}, Current best: {self.history.list_current_best[-1][self.ID_TAR][self.ID_FIT]}, "
              f"Global best: {self.history.list_global_best[-1][self.ID_TAR][self.ID_FIT]}, Runtime: {runtime:.5f} seconds")

    def track_optimize_process(self):
        """
        Save some historical data after training process finished
        """
        self.history.epoch = len(self.history.list_diversity)
        div_max = np.max(self.history.list_diversity)
        self.history.list_exploration = 100 * (np.array(self.history.list_diversity) / div_max)
        self.history.list_exploitation = 100 - self.history.list_exploration
        self.history.list_global_best = self.history.list_global_best[1:]
        self.history.list_current_best = self.history.list_current_best[1:]
        self.solution = self.history.list_global_best[-1]

    def create_population(self, pop_size=None):
        """
        Args:
            pop_size (int): number of solutions

        Returns:
            list: population or list of solutions/agents
        """
        if pop_size is None:
            pop_size = self.pop_size
        pop = []
        if self.mode == "thread":
            with parallel.ThreadPoolExecutor(self.n_workers) as executor:
                list_executors = [executor.submit(self.create_solution, self.problem.lb, self.problem.ub) for _ in range(pop_size)]
                # This method yield the result everytime a thread finished their job (not by order)
                for f in parallel.as_completed(list_executors):
                    pop.append(f.result())
        elif self.mode == "process":
            with parallel.ProcessPoolExecutor(self.n_workers) as executor:
                list_executors = [executor.submit(self.create_solution, self.problem.lb, self.problem.ub) for _ in range(pop_size)]
                # This method yield the result everytime a cpu finished their job (not by order).
                for f in parallel.as_completed(list_executors):
                    pop.append(f.result())
        else:
            pop = [self.create_solution(self.problem.lb, self.problem.ub) for _ in range(0, pop_size)]
        return pop

    def update_target_wrapper_population(self, pop=None):
        """
        Update target wrapper for input population

        Args:
            pop (list): the population

        Returns:
            list: population with updated fitness value
        """
        pos_list = [agent[self.ID_POS] for agent in pop]
        if self.mode == "thread":
            with parallel.ThreadPoolExecutor(self.n_workers) as executor:
                list_results = executor.map(self.get_target_wrapper, pos_list)  # Return result not the future object
                for idx, target in enumerate(list_results):
                    pop[idx][self.ID_TAR] = target
        elif self.mode == "process":
            with parallel.ProcessPoolExecutor(self.n_workers) as executor:
                list_results = executor.map(self.get_target_wrapper, pos_list)  # Return result not the future object
                for idx, target in enumerate(list_results):
                    pop[idx][self.ID_TAR] = target
        elif self.mode == "swarm":
            for idx, pos in enumerate(pos_list):
                pop[idx][self.ID_TAR] = self.get_target_wrapper(pos)
        return pop

    def get_global_best_solution(self, pop: list):
        """
        Sort population and return the sorted population and the best solution

        Args:
            pop (list): The population of pop_size individuals

        Returns:
            Sorted population and global best solution
        """
        sorted_pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT])  # Already returned a new sorted list
        if self.problem.minmax == "min":
            return sorted_pop, deepcopy(sorted_pop[0])
        else:
            return sorted_pop, deepcopy(sorted_pop[-1])

    def get_better_solution(self, agent1: list, agent2: list):
        """
        Args:
            agent1 (list): A solution
            agent2 (list): Another solution

        Returns:
            The better solution between them
        """
        if self.problem.minmax == "min":
            if agent1[self.ID_TAR][self.ID_FIT] < agent2[self.ID_TAR][self.ID_FIT]:
                return deepcopy(agent1)
            return deepcopy(agent2)
        else:
            if agent1[self.ID_TAR][self.ID_FIT] < agent2[self.ID_TAR][self.ID_FIT]:
                return deepcopy(agent2)
            return deepcopy(agent1)

    def compare_agent(self, agent_new: list, agent_old: list):
        """
        Args:
            agent_new (list): The new solution
            agent_old (list): The old solution

        Returns:
            boolean: Return True if the new solution is better than the old one and otherwise
        """
        if self.problem.minmax == "min":
            if agent_new[self.ID_TAR][self.ID_FIT] < agent_old[self.ID_TAR][self.ID_FIT]:
                return True
            return False
        else:
            if agent_new[self.ID_TAR][self.ID_FIT] < agent_old[self.ID_TAR][self.ID_FIT]:
                return False
            return True

    def get_special_solutions(self, pop=None, best=3, worst=3):
        """
        Args:
            pop (list): The population
            best (int): Top k1 best solutions, default k1=3, good level reduction
            worst (int): Top k2 worst solutions, default k2=3, worst level reduction

        Returns:
            list: sorted_population, k1 best solutions and k2 worst solutions
        """
        if self.problem.minmax == "min":
            pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT])
        else:
            pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT], reverse=True)
        if best is None:
            if worst is None:
                exit(0)
            else:
                return pop, None, deepcopy(pop[::-1][:worst])
        else:
            if worst is None:
                return pop, deepcopy(pop[:best]), None
            else:
                return pop, deepcopy(pop[:best]), deepcopy(pop[::-1][:worst])

    def get_special_fitness(self, pop=None):
        """
        Args:
            pop (list): The population

        Returns:
            list: Total fitness, best fitness, worst fitness
        """
        total_fitness = np.sum([agent[self.ID_TAR][self.ID_FIT] for agent in pop])
        if self.problem.minmax == "min":
            pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT])
        else:
            pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT], reverse=True)
        return total_fitness, pop[0][self.ID_TAR][self.ID_FIT], pop[-1][self.ID_TAR][self.ID_FIT]

    def update_global_best_solution(self, pop=None, save=True):
        """
        Update the global best solution saved in variable named: self.history_list_g_best

        Args:
            pop (list): The population of pop_size individuals
            save (bool): True if you want to add new current/global best to history, False if you just want to update current/global best

        Returns:
            list: Sorted population and the global best solution
        """
        if self.problem.minmax == "min":
            sorted_pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT])
        else:
            sorted_pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT], reverse=True)
        current_best = sorted_pop[0]
        if save:
            self.history.list_current_best.append(current_best)
            better = self.get_better_solution(current_best, self.history.list_global_best[-1])
            self.history.list_global_best.append(better)
            return deepcopy(sorted_pop), deepcopy(better)
        else:
            local_better = self.get_better_solution(current_best, self.history.list_current_best[-1])
            self.history.list_current_best[-1] = local_better
            global_better = self.get_better_solution(current_best, self.history.list_global_best[-1])
            self.history.list_global_best[-1] = global_better
            return deepcopy(sorted_pop), deepcopy(global_better)

    ## Selection techniques
    def get_index_roulette_wheel_selection(self, list_fitness: np.array):
        """
        This method can handle min/max problem, and negative or positive fitness value.

        Args:
            list_fitness (nd.array): 1-D numpy array

        Returns:
            int: Index of selected solution
        """
        scaled_fitness = (list_fitness - np.min(list_fitness)) / (np.ptp(list_fitness) + self.EPSILON)
        if self.problem.minmax == "min":
            final_fitness = 1.0 - scaled_fitness
        else:
            final_fitness = scaled_fitness
        total_sum = sum(final_fitness)
        r = np.random.uniform(low=0, high=total_sum)
        for idx, f in enumerate(final_fitness):
            r = r + f
            if r > total_sum:
                return idx
        return np.random.choice(range(0, len(list_fitness)))

    def get_index_kway_tournament_selection(self, pop=None, k_way=0.2, output=2, reverse=False):
        """
        Args:
            pop: The population
            k_way (float/int): The percent or number of solutions are randomized pick
            output (int): The number of outputs
            reverse (bool): set True when finding the worst fitness

        Returns:
            list: List of the selected indexes
        """
        if 0 < k_way < 1:
            k_way = int(k_way * len(pop))
        list_id = np.random.choice(range(len(pop)), k_way, replace=False)
        list_parents = [[idx, pop[idx][self.ID_TAR][self.ID_FIT]] for idx in list_id]
        if self.problem.minmax == "min":
            list_parents = sorted(list_parents, key=lambda agent: agent[1])
        else:
            list_parents = sorted(list_parents, key=lambda agent: agent[1], reverse=True)
        if reverse:
            return [parent[0] for parent in list_parents[-output:]]
        return [parent[0] for parent in list_parents[:output]]

    def get_levy_flight_step(self, beta=1.0, multiplier=0.001, size=None, case=0):
        """
        Get the Levy-flight step size

        Args:
            beta (float): Should be in range [0, 2].

                * 0-1: small range --> exploit
                * 1-2: large range --> explore

            multiplier (float): default = 0.001
            size (tuple, list): size of levy-flight steps, for example: (3, 2), 5, (4, )
            case (int): Should be one of these value [0, 1, -1].

                * 0: return multiplier * s * np.random.uniform()
                * 1: return multiplier * s * np.random.normal(0, 1)
                * -1: return multiplier * s

        Returns:
            int: The step size of Levy-flight trajectory
        """
        # u and v are two random variables which follow np.random.normal distribution
        # sigma_u : standard deviation of u
        sigma_u = np.power(gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)), 1 / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        size = 1 if size is None else size
        u = np.random.normal(0, sigma_u ** 2, size)
        v = np.random.normal(0, sigma_v ** 2, size)
        s = u / np.power(np.abs(v), 1 / beta)
        if case == 0:
            step = multiplier * s * np.random.uniform()
        elif case == 1:
            step = multiplier * s * np.random.normal(0, 1)
        else:
            step = multiplier * s
        return step[0] if size == 1 else step

    def levy_flight(self, epoch=None, position=None, g_best_position=None, step=0.001, case=0):
        """
        Get the Levy-flight position of current agent

        Args:
            epoch (int): The current epoch/iteration
            position: The position of current agent
            g_best_position: The position of the global best solution
            step (float): The step size in Levy-flight, default = 0.001
            case (int): Should be one of these value [0, 1, 2]

        Returns:
            The Levy-flight position of current agent
        """
        beta = 1
        # muy and v are two random variables which follow np.random.normal distribution
        # sigma_muy : standard deviation of muy
        sigma_muy = np.power(gamma(1 + beta) * np.sin(np.pi * beta / 2) / (gamma((1 + beta) / 2) * beta * np.power(2, (beta - 1) / 2)), 1 / beta)
        # sigma_v : standard deviation of v
        sigma_v = 1
        muy = np.random.normal(0, sigma_muy ** 2)
        v = np.random.normal(0, sigma_v ** 2)
        s = muy / np.power(np.abs(v), 1 / beta)
        levy = step * s * (g_best_position - position)

        if case == 0:
            return levy
        elif case == 1:
            return position + levy
        elif case == 2:
            return position + 1.0 / np.sqrt(epoch + 1) * np.sign(np.random.random(self.problem.n_dims) - 0.5) * levy
        elif case == 3:
            return g_best_position + levy
        else:
            return g_best_position + 1.0 / np.sqrt(epoch + 1) * levy

    ### Survivor Selection
    def greedy_selection_population(self, pop_old=None, pop_new=None):
        """
        Args:
            pop_old (list): The current population
            pop_new (list): The next population

        Returns:
            The new population with better solutions
        """
        len_old, len_new = len(pop_old), len(pop_new)
        if len_old != len_new:
            self.logger.error("Greedy selection of two population with different length.")
            exit(0)
        if self.problem.minmax == "min":
            return [pop_new[i] if pop_new[i][self.ID_TAR][self.ID_FIT] < pop_old[i][self.ID_TAR][self.ID_FIT]
                    else pop_old[i] for i in range(len_old)]
        else:
            return [pop_new[i] if pop_new[i][self.ID_TAR] > pop_old[i][self.ID_TAR]
                    else pop_old[i] for i in range(len_old)]

    def get_sorted_strim_population(self, pop=None, pop_size=None, reverse=False):
        """
        Args:
            pop (list): The population
            pop_size (int): The number of population
            reverse (bool): False (ascending fitness order), and True (descending fitness order)

        Returns:
            The sorted population with pop_size size
        """
        if self.problem.minmax == "min":
            pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT], reverse=reverse)
        else:
            pop = sorted(pop, key=lambda agent: agent[self.ID_TAR][self.ID_FIT], reverse=reverse)
        return pop[:pop_size]

    def create_opposition_position(self, agent=None, g_best=None):
        """
        Args:
            agent: The current solution (agent)
            g_best: the global best solution (agent)

        Returns:
            The opposite position
        """
        return self.problem.lb + self.problem.ub - g_best[self.ID_POS] + np.random.uniform() * (g_best[self.ID_POS] - agent[self.ID_POS])

    ### Crossover
    def crossover_arithmetic(self, dad_pos=None, mom_pos=None):
        """
        Args:
            dad_pos: position of dad
            mom_pos: position of mom

        Returns:
            list: position of 1st and 2nd child
        """
        r = np.random.uniform()  # w1 = w2 when r =0.5
        w1 = np.multiply(r, dad_pos) + np.multiply((1 - r), mom_pos)
        w2 = np.multiply(r, mom_pos) + np.multiply((1 - r), dad_pos)
        return w1, w2

    #### Improved techniques can be used in any algorithms: 1
    ## Based on this paper: An efficient equilibrium optimizer with mutation strategy for numerical optimization (but still different)
    ## This scheme used after the original and including 4 step:
    ##  s1: sort population, take p1 = 1/2 best population for next round
    ##  s2: do the mutation for p1, using greedy method to select the better solution
    ##  s3: do the search mechanism for p1 (based on global best solution and the updated p1 above), to make p2 population
    ##  s4: construct the new population for next generation
    def improved_ms(self, pop=None, g_best=None):  ## m: mutation, s: search
        pop_len = int(len(pop) / 2)
        ## Sort the updated population based on fitness
        pop = sorted(pop, key=lambda item: item[self.ID_TAR][self.ID_FIT])
        pop_s1, pop_s2 = pop[:pop_len], pop[pop_len:]

        ## Mutation scheme
        pop_new = []
        for i in range(0, pop_len):
            agent = deepcopy(pop_s1[i])
            pos_new = pop_s1[i][self.ID_POS] * (1 + np.random.normal(0, 1, self.problem.n_dims))
            agent[self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append(agent)
        pop_new = self.update_target_wrapper_population(pop_new)
        pop_s1 = self.greedy_selection_population(pop_s1, pop_new)  ## Greedy method --> improved exploitation

        ## Search Mechanism
        pos_s1_list = [item[self.ID_POS] for item in pop_s1]
        pos_s1_mean = np.mean(pos_s1_list, axis=0)
        pop_new = []
        for i in range(0, pop_len):
            agent = deepcopy(pop_s2[i])
            pos_new = (g_best[self.ID_POS] - pos_s1_mean) - np.random.random() * \
                      (self.problem.lb + np.random.random() * (self.problem.ub - self.problem.lb))
            agent[self.ID_POS] = self.amend_position(pos_new, self.problem.lb, self.problem.ub)
            pop_new.append(agent)
        ## Keep the diversity of populatoin and still improved the exploration
        pop_s2 = self.update_target_wrapper_population(pop_new)
        pop_s2 = self.greedy_selection_population(pop_s2, pop_new)

        ## Construct a new population
        pop = pop_s1 + pop_s2
        return pop
