from mealpy.evolutionary_based import DE
from mealpy.evolutionary_based.DE import L_SHADE
from mealpy.evolutionary_based.DE import SADE
import numpy as np
import csv
import objective_function as of

if __name__ == '__main__':
    algoname = 'de'
    problems = ["Node400", "Node500", "Node600", "Node700"]
    num_run = 2
    for i in problems:
        if i == "Node200":
            fitness_function = of.NODEE200().evaluate
            dim_ = len(of.NODEE200().Candidate_List)
            fname = 'Node200result_' + algoname + '.csv'
        elif i == "Node300":
            fitness_function = of.NODEE300().evaluate
            dim_ = len(of.NODEE300().Candidate_List)
            fname = 'Node300result_' + algoname + '.csv'
        elif i == "Node400":
            fitness_function = of.NODEE400().evaluate
            dim_ = len(of.NODEE400().Candidate_List)
            fname = 'Node400result_' + algoname + '.csv'
        elif i == "Node500":
            fitness_function = of.NODEE500().evaluate
            dim_ = len(of.NODEE500().Candidate_List)
            fname = 'Node500result_' + algoname + '.csv'
        elif i == "Node600":
            fitness_function = of.NODEE600().evaluate
            dim_ = len(of.NODEE600().Candidate_List)
            fname = 'Node600result_' + algoname + '.csv'
        elif i == "Node700":
            fitness_function = of.NODEE700().evaluate
            dim_ = len(of.NODEE700().Candidate_List)
            fname = 'Node700result_' + algoname + '.csv'
        
        f2 = open(fname, 'w')
        writer = csv.writer(f2)
        
        problem = {
            "fit_func": fitness_function,
            "lb": 0,
            "ub": 1,
            "minmax": "min",
            "n_dims": dim_,
            "save_population": False,
            "log_to": None,  # console, file "log_file": "result.log"
        }

        term_dict1 = {
            "mode": "FE",
            "quantity": 50000  # 100000 number of function evaluation
        }
        for j in range(0, num_run):
            dd = [i, j]
            print(f"Algorithm: {algoname}, run: {dd}")
            model = DE.BaseDE(problem, epoch=100, pop_size=100, wf=0.8, cr=0.2, strategy=1, termination=term_dict1)
            # model = L_SHADE(problem, epoch=50, pop_size=10, miu_f=0.5, miu_cr=0.5, termination=term_dict1)
            # model = SADE(problem, epoch=50, pop_size=10, termination=term_dict1)
            best_position, best_fitness = model.solve()
            time = sum(model.history.list_epoch_time)
            # print(f"Best solution: {model.solution}, Best fitness: {best_fitness}")

            dd = dd + [best_fitness, time]
            writer.writerow(dd)
            print(f"output: {dd}")
        f2.close()

