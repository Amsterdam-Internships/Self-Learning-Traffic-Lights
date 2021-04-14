import pandas as pd
import cityflow

from src.utility import *


"""
This file evaluates the chosen actions with the signal_plan_template.txt file.

Source: https://github.com/tianrang-intelligence/TSCC2019
"""

with open('src/config.json') as json_file:
    config2 = json.load(json_file)


def main():
    args = parse_arguments()
    # sim_setting = config2
    # sim_setting["num_step"] = 300
    # evaluate_one_traffic(sim_setting, args.scenario)


def evaluate_one_traffic(config, printing='no_printing'):
    args = parse_arguments()

    # Plan files should be named after signal_plan_template + the intersection_index.
    plan_files = [[] for i in config['intersection_indices']]
    for i, intersection_index in enumerate(config['intersection_indices']):
        plan_files[i] = config['path_save'] + '/signal_plan_template' + str(intersection_index) + '.txt'
    out_file = config['path_save'] + '/evaluation.txt'
    out_file2= config['path_save'] + '/travel_time_data.json'

    tt = 0
    if check(plan_files[0], config["num_step"]):  # Checks only the first file.
        tt, tt_list = cal_travel_time(config, plan_files)
        if printing == 'print':
            print("")
            print("====================== travel time ======================")
            print(config['mode'] + ": scenario_{0}: {1:.2f} s".format(config['scenario'], tt))

            with open(out_file, "w") as f:
                f.write(str(tt))

            save_pickle(tt_list, out_file2)
    else:
        print("planFile is invalid, Rejected!")

    return tt


def cal_travel_time(dic_sim_setting, plan_files):
    dic_sim_setting['saveReplay'] = True

    path = "src/config_{}_args2.json".format(dic_sim_setting['scenario'])
    with open(path, 'w') as outfile:
        json.dump(dic_sim_setting, outfile)
    eng = cityflow.Engine(path, thread_num=1)

    plans = [[] for plan in plan_files]
    intersection_ids = [0 for plan in plan_files]
    for i in range(len(plan_files)):
        plans[i] = pd.read_csv(plan_files[i], sep="\t", header=0, dtype=int)
        intersection_ids[i] = plans[i].columns[0]

    tt_list = []
    for step in range(dic_sim_setting["num_step"]):
        for i in range(len(plans)):
            phase = int(plans[i].loc[step])
            eng.set_tl_phase(intersection_ids[i], phase)
        eng.next_step()
        tt_list.append(eng.get_average_travel_time())

    return eng.get_average_travel_time(), tt_list


def check(plan_file, num_step):
    flag = True
    error_info = ''
    try:
        plan = pd.read_csv(plan_file, sep='\t', header=0, dtype=int)
    except:
        flag = False
        error_info = 'The format of signal plan is not valid and cannot be read by pd.read_csv!'
        print(error_info)
        return flag

    intersection_id = plan.columns[0]
    # if intersection_id != 'intersection_1_1':
    #     flag = False
    #     error_info = 'The header intersection_id is wrong (for example: intersection_1_1)!'
    #     print(error_info)
    #     return flag

    phases = plan.values
    current_phase = phases[0][0]

    if len(phases) < num_step:
        flag = False
        error_info = 'The time of signal plan is less than the default time!'
        print(error_info)
        return flag

    if current_phase == 0:
        yellow_time = 1
    else:
        yellow_time = 0

    # get first green phase and check
    last_green_phase = '*'
    for next_phase in phases[1:]:
        next_phase = next_phase[0]

        # check phase itself
        if next_phase == '':
            continue
        # if next_phase not in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        if next_phase not in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]:
            flag = False
            # error_info = 'Phase must be in [0, 1, 2, 3, 4, 5, 6, 7, 8]!'
            error_info = 'Phase must be in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]!'
            break

        # check changing phase
        if next_phase != current_phase and next_phase != 0 and current_phase != 0:
            flag = False
            error_info = '5 seconds of yellow time must be inserted between two different phase!'
            break

        # # check unchangeable phase
        # if next_phase != 0 and next_phase == last_green_phase:
        #     flag = False
        #     error_info = 'No yellow light is allowed between the same phase!'
        #     break

        # check yellow time
        if next_phase != 0 and yellow_time != 0 and yellow_time != 5:
            flag = False
            error_info = 'Yellow time must be 5 seconds!'
            break

        # normal
        if next_phase == 0:
            yellow_time += 1
            if current_phase != 0:
                last_green_phase = current_phase
        else:
            yellow_time = 0
        current_phase = next_phase

    if not flag:
        print(error_info)
    return flag


if __name__ == "__main__":
    main()
