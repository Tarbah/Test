import pickle
import matplotlib.pyplot as plt
import ast
import os
import numpy as np


results = list()
max_len_hist = 0
max_time_steps = 0
AGA_errors = list()
ABU_errors = list()
PF_errors = list()

AGA_timeSteps = list()
ABU_timeSteps = list()
PF_timeSteps = list()

AGA_estimationHistError = []

AGA_estimationHist = list()
ABU_estimationHist = list()
PF_estimationHist = list()

O_AGA_comp_time = list()
M_AGA_comp_time = list()

O_ABU_comp_time = list()
M_ABU_comp_time = list()
PF_comp_time = list()

O_AGA_timeSteps = []
M_AGA_timeSteps = []

O_ABU_timeSteps = []
M_ABU_timeSteps = []

O_PF_timeSteps = []
M_PF_timeSteps = []


ave_aga_levels = []
ave_aga_angle = []
ave_aga_radius = []
ave_abu_levels = []
ave_abu_angle = []
ave_abu_radius = []
ave_pf_levels = []
ave_pf_angle = []
ave_pf_radius = []

ust_info = []


def read_data_for_UCT():
    for root, dirs, files in os.walk('output16'):
        if 'pickleResults.txt' in files:
            print root
            with open(os.path.join(root,'pickleResults.txt'), "r") as pickleFile:

                UCT_Dictionary = {}
                dataList = pickle.load(pickleFile)

                data = dataList[1]
                systemDetails = dataList[0]

                # Simulator Information
                UCT_Dictionary['simWidth'] = systemDetails['simWidth']
                UCT_Dictionary['simHeight'] = systemDetails['simHeight']
                UCT_Dictionary['agentsCounts'] = systemDetails['agentsCounts']
                UCT_Dictionary['itemsCounts'] = systemDetails['itemsCounts']
                UCT_Dictionary['iterationMax'] = systemDetails['iterationMax']
                UCT_Dictionary['maxDepth'] = systemDetails['maxDepth']
                UCT_Dictionary['mcts_mode'] = systemDetails['mcts_mode']

                beginTime = systemDetails['beginTime']
                endTime = systemDetails['endTime']
                UCT_Dictionary['computationalTime'] = int(endTime) - int(beginTime)
                UCT_Dictionary['estimationMode'] = systemDetails['estimationMode']
                UCT_Dictionary['timeSteps'] = systemDetails['timeSteps']
                ust_info.append(UCT_Dictionary)


def extract_uct_inf():
    global ust_info
    global max_time_steps
    max_time_steps = 0
    for result in ust_info:
        if int(result['timeSteps']) > max_time_steps :
            max_time_steps = int( result['timeSteps'])

        if result['estimationMode'] == 'AGA':

            if result['mcts_mode'] == 'MSPA':
                M_AGA_timeSteps.append(result['timeSteps'])
                M_AGA_comp_time.append(result['computationalTime'])
            else:
                O_AGA_timeSteps.append(result['timeSteps'])
                O_AGA_comp_time.append(result['computationalTime'])

        if result['estimationMode'] == 'ABU':

            if result['mcts_mode'] == 'MSPA':
                M_ABU_timeSteps.append(result['timeSteps'])
                M_ABU_comp_time.append(result['computationalTime'])
            else:
                O_ABU_timeSteps.append(result['timeSteps'])
                O_ABU_comp_time.append(result['computationalTime'])


def read_files():
    for root, dirs, files in os.walk('Output_2agents_size10'):
        if 'pickleResults.txt' in files:
            print root
            with open(os.path.join(root,'pickleResults.txt'),"r") as pickleFile:

                estimationDictionary = {}
                dataList = pickle.load(pickleFile)

                data = dataList[1]
                systemDetails = dataList[0]

                # Simulator Information
                simWidth = systemDetails['simWidth']
                simHeight = systemDetails['simHeight']
                agentsCounts = systemDetails['agentsCounts']
                itemsCounts = systemDetails['itemsCounts']
                iterationMax = systemDetails['iterationMax']
                maxDepth = systemDetails['maxDepth']
                generatedDataNumber = systemDetails['generatedDataNumber']
                reuseTree = systemDetails['reuseTree']
                mcts_mode = systemDetails['mcts_mode']

                estimationDictionary['typeSelectionMode'] = systemDetails['typeSelectionMode']

                beginTime = systemDetails['beginTime']
                endTime = systemDetails['endTime']

                estimationDictionary['computationalTime'] = int(endTime) - int(beginTime)
                estimationDictionary['estimationMode'] = systemDetails['estimationMode']
                estimationDictionary['timeSteps'] = systemDetails['timeSteps']
                estimationDictionary['mcts_mode'] = systemDetails['mcts_mode']

                agentDictionary = data[0]
                selected_type_prob = []
                historyParameters = [[]]

                maxProbability = max(agentDictionary['l1LastProbability'],agentDictionary['l2LastProbability'],
                                     agentDictionary['f1LastProbability'],agentDictionary['f2LastProbability'])

                estimationDictionary['estimatedType'] = maxProbability

                if maxProbability == agentDictionary['l1LastProbability']:
                    selected_type_prob = agentDictionary['l1']
                    historyParameters = ast.literal_eval(agentDictionary['l1EstimationHistory'])

                elif maxProbability == agentDictionary['l2LastProbability']:
                    selected_type_prob = agentDictionary['l2']
                    historyParameters = ast.literal_eval(agentDictionary['l2EstimationHistory'])

                elif maxProbability == agentDictionary['f1LastProbability']:
                    selected_type_prob = agentDictionary['f1']
                    historyParameters = ast.literal_eval(agentDictionary['f1EstimationHistory'])

                else:
                    selected_type_prob = agentDictionary['f2']
                    historyParameters = ast.literal_eval(agentDictionary['f2EstimationHistory'])

                estimationDictionary['last_estimated_value'] = historyParameters[len(historyParameters)-1]

                trueParameters = agentDictionary['trueParameters']

                estimationDictionary['historyParameters'] = historyParameters

                print  estimationDictionary['estimationMode']
                last_estimated_value = [estimationDictionary['last_estimated_value'][0] , estimationDictionary['last_estimated_value'][1],estimationDictionary['last_estimated_value'][2]]

                diff = [x - y for x, y in zip(trueParameters, last_estimated_value)]

                estimationDictionary['estimationError'] = diff

                results.append(estimationDictionary)
    return results

def extract_information():
    global results
    global max_len_hist
    for result in results:

        if max_len_hist < len(result['historyParameters']):
            max_len_hist = len(result['historyParameters'])

        if result['estimationMode'] == 'AGA':
            AGA_errors.append(result['estimationError'])

            AGA_estimationHist.append(result['historyParameters'])
            AGA_timeSteps.append(result['timeSteps'])


            if result['mcts_mode']=='MSPA':
                M_AGA_timeSteps.append(result['timeSteps'])
                M_AGA_comp_time.append(result['computationalTime'])
            else:
                O_AGA_timeSteps.append(result['timeSteps'])
                O_AGA_comp_time.append(result['computationalTime'])

        if result['estimationMode'] == 'ABU':
            ABU_errors.append(result['estimationError'])
            # ABU_estimationHistError.append(result['estimationHistError'])
            ABU_estimationHist.append(result['historyParameters'])
            ABU_timeSteps.append(result['timeSteps'])


            if result['mcts_mode']=='MSPA':
                M_ABU_timeSteps.append(result['timeSteps'])
                M_ABU_comp_time.append(result['computationalTime'])
            else:
                O_ABU_timeSteps.append(result['timeSteps'])
                O_ABU_comp_time.append(result['computationalTime'])

        if result['estimationMode'] == 'PF':
            PF_errors.append(result['estimationError'])
            # PF_estimationHistError.append(result['estimationHistError'])
            PF_timeSteps.append(result['timeSteps'])
            PF_estimationHist.append(result['historyParameters'])
            PF_comp_time.append(result['computationalTime'])

            if result['mcts_mode'] == 'MSPA':
                M_PF_timeSteps.append(result['timeSteps'])
            else:
                O_PF_timeSteps.append(result['timeSteps'])

# Normalizing history
def plot_history_of_estimation():
    global ave_aga_levels
    global ave_aga_angle
    global ave_aga_radius
    global ave_abu_levels
    global ave_abu_angle
    global ave_abu_radius
    global ave_pf_levels
    global ave_pf_angle
    for a in AGA_estimationHist:

        last_value = a[len(a)-1]
        diff = max_len_hist - len(a)
        for i in range(diff):
            a.append(last_value)

    aa = np.array(AGA_estimationHist)
    ave = aa.mean(0)
    a_data_set = np.transpose(ave)
    ave_aga_levels = list(a_data_set[0, :])
    ave_aga_angle = list(a_data_set[1, :])
    ave_aga_radius = list(a_data_set[2, :])

    for a in ABU_estimationHist:
        last_value = a[len(a) - 1]
        diff = max_len_hist - len(a)
        for i in range(diff):
            a.append(last_value)

    aa = np.array(ABU_estimationHist)
    ave = aa.mean(0)
    a_data_set = np.transpose(ave)
    ave_abu_levels = a_data_set[0, :]
    ave_abu_angle = a_data_set[1, :]
    ave_abu_radius = a_data_set[2, :]

    for a in PF_estimationHist:
        last_value = a[len(a) - 1]
        diff = max_len_hist - len(a)
        for i in range(diff):
            a.append(last_value)

    aa = np.array(PF_estimationHist)
    ave = aa.mean(0)
    a_data_set = np.transpose(ave)
    ave_pf_levels = a_data_set[0, :]
    ave_pf_angle = a_data_set[1, :]
    ave_pf_radius = a_data_set[2, :]

    fig = plt.figure(2)

    plt.subplot(3,1,1)
    plt.plot([i for i in range(len(ave_pf_levels))], ave_pf_levels, label='PF', linestyle='-', color='cornflowerblue',linewidth=1)
    plt.plot([i for i in range(len(ave_abu_levels))], ave_abu_levels, label='ABU', linestyle='-', color='red',linewidth=1)
    plt.plot([i for i in range(len(ave_aga_levels))], ave_aga_levels, label='AGA', linestyle='-', color='green',linewidth=1)
    ax = plt.gca()
    ax.set_ylabel('Level ')
    ax.legend(loc="upper center", shadow=False, fontsize='x-small')
    plt.subplot(3,1,2)
    plt.plot([i for i in range(len(ave_pf_angle))], ave_pf_angle, label='PF', linestyle='-', color='cornflowerblue',linewidth=1)
    plt.plot([i for i in range(len(ave_abu_angle))], ave_abu_angle, label='ABU', linestyle='-', color='red',linewidth=1)
    plt.plot([i for i in range(len(ave_aga_angle))], ave_aga_angle, label='AGA', linestyle='-', color='green',linewidth=1)
    ax = plt.gca()
    ax.set_ylabel('Angle ')

    plt.subplot(3,1,3)
    plt.plot([i for i in range(len(ave_pf_radius))], ave_pf_radius, label='PF', linestyle='-', color='cornflowerblue',linewidth=1)
    plt.plot([i for i in range(len(ave_abu_radius))], ave_abu_radius, label='ABU', linestyle='-', color='red',linewidth=1)
    plt.plot([i for i in range(len(ave_aga_radius))], ave_aga_radius, label='AGA', linestyle='-', color='green',linewidth=1)
    ax = plt.gca()
    ax.set_ylabel('Level ')
    ax.set_xlabel('Step numbers')
    plt.show()
    #fig.savefig("./plots/history_of_estimation.jpg")

def plot_errors_in_history_estimation(level,angle,radius):
    global ave_aga_levels
    global ave_aga_angle
    global ave_aga_radius
    global ave_abu_levels
    global ave_abu_angle
    global ave_abu_radius
    global ave_pf_levels
    global ave_pf_angle
    global ave_pf_levels

    err_aga_levels = []
    err_aga_angle = []
    err_aga_radius = []
    err_abu_levels = []
    err_abu_angle = []
    err_abu_radius = []
    err_pf_levels = []
    err_pf_angle = []
    err_pf_radius = []

    for i in range (max_len_hist):
        err_aga_levels.append( ave_aga_levels[i] - level)
        err_aga_angle.append(ave_aga_angle[i] - angle)
        err_aga_radius.append(ave_aga_radius[i] - radius)
        err_abu_levels.append(ave_abu_levels[i] - level)
        err_abu_angle.append(ave_abu_angle[i] - angle)
        err_abu_radius.append(ave_abu_radius[i] - radius)
        err_pf_levels.append(ave_pf_levels[i] - level)
        err_pf_angle.append(ave_pf_angle[i] - angle)
        err_pf_radius.append(ave_pf_radius[i] - radius)

        fig = plt.figure(1)

        plt.subplot(3,1,1)
        plt.plot([i for i in range(len(err_pf_levels))], err_pf_levels, label='PF', linestyle='-', color='cornflowerblue',linewidth=1)
        plt.plot([i for i in range(len(err_abu_levels))], err_abu_levels, label='ABU', linestyle='-', color='red',linewidth=1)
        plt.plot([i for i in range(len(err_aga_levels))], err_aga_levels, label='AGA', linestyle='-', color='green',linewidth=1)
        ax = plt.gca()
        ax.set_ylabel('Level Error')
        ax.legend(loc="upper right", shadow=True, fontsize='x-large')
        plt.subplot(3,1,2)
        plt.plot([i for i in range(len(err_pf_angle))], err_pf_angle, label='PF', linestyle='-', color='cornflowerblue',linewidth=1)
        plt.plot([i for i in range(len(err_abu_angle))], err_abu_angle, label='ABU', linestyle='-', color='red',linewidth=1)
        plt.plot([i for i in range(len(err_aga_angle))], err_aga_angle, label='AGA', linestyle='-', color='green',linewidth=1)
        ax = plt.gca()
        ax.set_ylabel('Angle Error')

        plt.subplot(3,1,3)
        plt.plot([i for i in range(len(err_pf_radius))], err_pf_radius, label='PF', linestyle='-', color='cornflowerblue',linewidth=1)
        plt.plot([i for i in range(len(err_abu_radius))], err_abu_radius, label='ABU', linestyle='-', color='red',linewidth=1)
        plt.plot([i for i in range(len(err_aga_radius))], err_aga_radius, label='AGA', linestyle='-', color='green',linewidth=1)
        ax = plt.gca()
        ax.set_ylabel('radius Error')
        ax.set_xlabel('Step numbers')
        plt.show()
        # fig.savefig("./plots/errors_in_history_estimation.jpg")

def plot_errors_in_last_estimation():
    AGA_ave_level = 0
    ABU_ave_level = 0
    PF_ave_level = 0

    if len(AGA_errors):
        AGA_data_set = np.transpose(np.array(AGA_errors))

        AGA_levels = AGA_data_set[0, :]
        AGA_ave_level = np.average(AGA_levels)

        AGA_angle = AGA_data_set[1, :]
        AGA_ave_angle = np.average(AGA_angle)

        AGA_radius = AGA_data_set[2, :]
        AGA_ave_radius = np.average(AGA_radius)

    if len(PF_errors):
        PF_data_set = np.transpose(np.array(PF_errors))

        PF_levels = PF_data_set[0, :]
        PF_ave_level = np.average(PF_levels)

        PF_angle = PF_data_set[1, :]
        PF_ave_angle = np.average(PF_angle)

        PF_radius = PF_data_set[2, :]
        PF_ave_radius = np.average(PF_radius)


    if len(ABU_errors):
        ABU_data_set = np.transpose(np.array(ABU_errors))

        ABU_levels = ABU_data_set [0, :]
        ABU_ave_level = np.average(ABU_levels)

        ABU_angle = ABU_data_set[1, :]
        ABU_ave_angle = np.average(ABU_angle)

        ABU_radius = ABU_data_set[2, :]
        ABU_ave_radius = np.average(ABU_radius)

    N = 3
    ind = np.arange(N)  # the x locations for the groups
    width = 0.20  # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)

    level_vals = [PF_ave_level, ABU_ave_level, AGA_ave_level]
    rects1 = ax.bar(ind, level_vals, width, color='r')
    angle_vals = [PF_ave_angle, ABU_ave_angle, AGA_ave_angle]
    rects2 = ax.bar(ind + width, angle_vals, width, color='g')
    radius_vals = [PF_ave_radius, ABU_ave_radius, AGA_ave_radius]
    rects3 = ax.bar(ind + width * 2, radius_vals, width, color='b')

    ax.set_title('Errors in Last estimation for different methods')
    ax.set_ylabel('Error')
    ax.set_xlabel('Estimation Method')
    ax.set_xticks(ind + width)
    ax.set_xticklabels(('PF', 'ABU', 'AGA'))
    ax.legend((rects1[0], rects2[0], rects3[0]), ('level', 'angle', 'radius'))
    plt.show()
    # fig.savefig("./plots/errors_in_last_estimation.jpg")


def plot_MonteCarlo():

    N = 3
    ind = np.arange(N)  # the x locations for the groups
    width = 0.10       # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)


    OSPA = [np.mean(O_AGA_timeSteps),np.mean(O_ABU_timeSteps),np.mean(O_PF_timeSteps)]
    rects1 = ax.bar(ind, OSPA, width, color='r')


    MSPA = [np.mean(M_AGA_timeSteps),np.mean(M_ABU_timeSteps),np.mean(M_PF_timeSteps)]
    rects2 = ax.bar(ind+ width, MSPA, width, color='b')

    ax.set_title('MonteCarlo')
    ax.set_ylabel('Time Steps')
    ax.set_xlabel('Estimation Method')
    ax.set_xticks(ind+width)
    ax.set_xticklabels(('AGA','ABU','PF'))
    ax.legend((rects1[0], rects2[0]), ('One State Per Action','Multiple State Per Action'))
    plt.show()
    # fig.savefig("./plots/MonteCarlo.jpg")


def plot_MonteCarlo_time():

    N = 2
    ind = np.arange(N)  # the x locations for the groups
    width = 0.10       # the width of the bars

    fig = plt.figure()
    ax = fig.add_subplot(111)


    OSPA = [np.mean(O_AGA_comp_time),np.mean(O_ABU_comp_time)] # ,np.mean(O_PF_timeSteps)]
    rects1 = ax.bar(ind, OSPA, width, color='r')


    MSPA = [np.mean(M_AGA_comp_time),np.mean(M_ABU_comp_time)]#,np.mean(M_PF_timeSteps)]
    rects2 = ax.bar(ind+ width, MSPA, width, color='b')


    ax.set_title('MonteCarlo')
    ax.set_ylabel('Computational Time')
    ax.set_xlabel('Estimation Method')
    ax.set_xticks(ind+width)
    ax.set_xticklabels(('AGA','ABU')) #,'PF'))
    ax.legend((rects1[0], rects2[0]), ('One State Per Action','Multiple State Per Action'))
    plt.show()
    # fig.savefig("./plots/MonteCarlo.jpg")

def multiple_agents():
    global ust_info

    algorithms = ["uct", "uct-h"]
    algorithmsLabels = ["UCT", "UCT-H"]

    algorithmsSymbol = ["o", "v"]

    nAgents = [1, 2, 3]
    max_time_steps = 6

    # nSamples = 1

    data = np.zeros((len(nAgents), len(algorithms), len(range(max_time_steps))))
    count = [0,0,0,0,0,0]
    print data

    dataMean = np.zeros((len(nAgents), len(algorithms)))
    dataUpCi = np.zeros((len(nAgents), len(algorithms)))

    u = 0
    u_data = [[]] * len(nAgents)
    h_data = [[]] * len(nAgents)

    for u in ust_info:
        agent_count = int(u["agentsCounts"])
        print agent_count
        print u['timeSteps']


        if u["mcts_mode"]=='MSPA':
            index = count[(agent_count-1) * 2]
            data[agent_count-1,0,index] = u['timeSteps']
            count[(agent_count - 1) * 2] +=1

        if u["mcts_mode"]=='OSPA':
            index = count[((agent_count - 1) * 2) + 1]
            data[agent_count-1,1,index] = u['timeSteps']
            count[((agent_count - 1) * 2) + 1] +=1

    print data[0, 0, 0:count[(0 * 2) + 0]]
    ro = ""

    # for u in
    # #
    stddev = np.std(data, ddof=1)
    for n in range(len(nAgents)):

        for a in range(len(algorithms)):
            print data[n, a, :]
            dataMean[n, a] = np.mean(data[n, a, 0:count[(n* 2) + a ]])
            dataUpCi[n, a] = np.std(data[n, a, 0:count[(n* 2) + a ]], ddof=1)
    # #
    plt.figure(figsize=(4, 3.0))

    for a in range(len(algorithms)):
         plt.errorbar(nAgents, dataMean[:, a], yerr=[m - n for m, n in zip(dataUpCi[:, a], dataMean[:, a])],
                      label=algorithmsLabels[a], marker=algorithmsSymbol[a])
         plt.errorbar(nAgents,dataMean[:,a], label=algorithmsLabels[a])

    plt.legend(loc=1)

    plt.xlim([0, 5])
    plt.xlabel("Number of Agents")
    plt.ylabel("Number of Iterations")

    plt.savefig("nAgents-15.pdf", bbox_inches='tight')


    plt.show()



# read_files()
# extract_information()
# plot_history_of_estimation()
# plot_errors_in_last_estimation()
# plot_errors_in_history_estimation()
read_data_for_UCT()
multiple_agents()
# plot_MonteCarlo()
# plot_MonteCarlo_time()

