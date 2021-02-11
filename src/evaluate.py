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
    sim_setting = config2
    sim_setting["num_step"] = 300
    evaluate_one_traffic(sim_setting, args.scenario)


def evaluate_one_traffic(config, scenario, mode='train', printing='no_printing'):
    args = parse_arguments()
    plan_file = "{}/experiments/{}/{}/{}/signal_plan_template.txt".format(args.output_dir, args.exp_name, config['mode'], config['hyperparams'])
    out_file = "{}/experiments/{}/{}/{}/evaluation.txt".format(args.output_dir, args.exp_name, config['mode'], config['hyperparams'])
    out_file2 = "{}/experiments/{}/{}/{}/travel_time_data.json".format(args.output_dir, args.exp_name, config['mode'], config['hyperparams'])

    if check(plan_file, config["num_step"]):
        tt, actions, tt_list = cal_travel_time(config, plan_file)
        if printing == 'print':
            print("")
            print("====================== travel time ======================")
            print(config['mode'] + ": scenario_{0}: {1:.2f} s".format(config['scenario'], tt))

            # change to baseline of fixed or sotl later. if score is > 0 you approved by that margin,
            # if score is <0 you got worse.
            # b = 58.88  # SOTL tt for 6.0.real_1x1_straight
            # b = 77.89  # SOTL average travel time for 1x1
            # b = 89.66  # SOTL tt 7.0.real_1x1_turns
            # score = (b - tt)/b
            #
            # print("====================== score ======================")
            # print("scenario_{0}: {1}".format(scenario, score))
            # print("====================== score ======================")

            with open(out_file, "w") as f:
                f.write(str(list(actions.values())) + '\n')
                f.write(str(tt))

            save_pickle(tt_list, out_file2)
    else:
        print("planFile is invalid, Rejected!")


def cal_travel_time(dic_sim_setting, plan_file):
    dic_sim_setting['saveReplay'] = True

    # Write to file so the engine can open it.
    # with open('src/config_args2.json', 'w') as outfile:
    #     json.dump(dic_sim_setting, outfile)

    if dic_sim_setting['data_set_mode'] == 'train':
        with open('src/config_args2.json', 'w') as outfile:
            json.dump(dic_sim_setting, outfile)
        eng = cityflow.Engine("src/config_args2.json", thread_num=1)
    if dic_sim_setting['data_set_mode'] == 'test':
        with open('src/config_args2_test.json', 'w') as outfile:
            json.dump(dic_sim_setting, outfile)
        eng = cityflow.Engine("src/config_args2_test.json", thread_num=1)

    # eng = cityflow.Engine("src/config_args2.json", thread_num=1)

    plan = pd.read_csv(plan_file, sep="\t", header=0, dtype=int)
    intersection_id = plan.columns[0]

    actions = {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
    tt_list = []

    for step in range(dic_sim_setting["num_step"]):
        phase = int(plan.loc[step])
        actions[phase-1] += 1
        tt_list.append(eng.get_average_travel_time())
        eng.set_tl_phase(intersection_id, phase)
        eng.next_step()

    return eng.get_average_travel_time(), actions, tt_list


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
    if intersection_id != 'intersection_1_1':
        flag = False
        error_info = 'The header intersection_id is wrong (for example: intersection_1_1)!'
        print(error_info)
        return flag

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
        if next_phase not in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
            flag = False
            error_info = 'Phase must be in [0, 1, 2, 3, 4, 5, 6, 7, 8]!'
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
