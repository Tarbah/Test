import random
from numpy.random import choice
import agent
from sklearn import linear_model
import numpy as np
import scipy.stats as st
from scipy import integrate
from copy import deepcopy
from copy import copy
import logging
import sys

logging.basicConfig(filename='parameter_estimation.log', format='%(asctime)s %(message)s', level=logging.DEBUG)


radius_max = 1
radius_min = 0.1
angle_max = 1
angle_min = 0.1
level_max = 1
level_min = 0

types = ['l1', 'l2', 'f1', 'f2']

class Parameter:
    def __init__(self, level, angle, radius):
        self.level = level
        self.angle = angle
        self.radius = radius
        self.iteration = 0
        self.min_max = [(0, 1), (0.1, 1), (0.1, 1)]

    def update(self, added_value):
        self.level += added_value[0]
        self.angle += added_value[1]
        self.radius += added_value[2]
        return self


########################################################################################################################
class TypeEstimation:
    def __init__(self, a_type):
        self.type = a_type
        self.type_probability = 0
        self.type_probabilities = []
        self.estimation_history = []
        self.action_probabilities = []
        self.internal_state = None
        self.data_set = []
        self.false_data_set = []
        self.choose_target_state = None
        # self.weight = []

    def add_estimation_history(self,probability, level, angle, radius):
        new_parameter = Parameter(level, angle, radius)
        self.estimation_history.append(new_parameter)
        self.type_probabilities.append(probability)

    def get_last_type_probability(self):
        return self.type_probabilities[len(self.type_probabilities)-1]

    def get_last_estimation(self):
        return self.estimation_history[len(self.estimation_history)-1]

    def update_estimation(self,estimation, action_probability):
        self.estimation_history.append(estimation)
        self.action_probabilities.append(action_probability)

    def get_value_for_update_belief(self):
        t = len(self.type_probabilities) - 1
        return self.type_probabilities[t - 1] * self.action_probabilities[t - 1]

    # def update_belief(self, belief_value):
    #     self.type_probabilities.append(belief_value)

    def get_estimation_history(self):
        estimation_historty = "["
        for est_hist in self.estimation_history:
            estimation_historty += "[" + str(est_hist.level) + "," + str(est_hist.angle) + "," + str(est_hist.radius) + "],"

        estimation_historty= estimation_historty [0:len(estimation_historty)-1]
        estimation_historty += "]"
        return estimation_historty


########################################################################################################################
class ParameterEstimation:

    def __init__(self):

        # P(teta|H)
        self.l1_estimation = TypeEstimation('l1')
        self.l2_estimation = TypeEstimation('l2')
        self.f1_estimation = TypeEstimation('f1')
        self.f2_estimation = TypeEstimation('f2')
        self.sim = None
        self.estimated_agent = None
        self.action_step_num = 0
        # type_selection_mode are: all types selection 'AS', Posterior Selection 'PS' , Bandit Selection 'BS'
        self.type_selection_mode = None
        self.load_states = []
        # Parameter estimation mode is AGA if it is Approximate Gradient Ascent ,
        # ABU if it is Approximate Bayesian Updating
        self.parameter_estimation_mode = None
        self.generated_data_number = None
        self.polynomial_degree = None
        self.iteration = 0

        ## Prior belief for ABU method
        self.belief_poly = [None]*3

    ####################################################################################################################
    # Initialisation random values for parameters of each type and probability of actions in time step 0

    def estimation_configuration(self, type_selection_mode, parameter_estimation_mode, generated_data_number,
                                 polynomial_degree, PF_add_threshold,PF_del_threshold,PF_weight):

        # type_selection_mode are: all types selection 'AS', Posterior Selection 'PS' , Bandit Selection 'BS'
        self.type_selection_mode = type_selection_mode

        # Parameter estimation mode is AGA if it is Approximate Gradient Ascent ,
        # ABU if it is Approximate Bayesian Updating
        self.parameter_estimation_mode = parameter_estimation_mode

        # the number of data we want to generate for estimating
        self.generated_data_number = generated_data_number
        self.polynomial_degree = polynomial_degree

        self.PF_add_threshold = PF_add_threshold
        self.PF_del_threshold = PF_del_threshold
        self.PF_weight = PF_weight

    ####################################################################################################################
    ####################################################################################################################
    # Initialisation random values for parameters of each type and probability of actions in time step 0
    def estimation_initialisation(self):
        # # P(teta|H) in t = 0

        # l1_init_prob = round(random.uniform(0, 6AGA_O_2), 6AGA_O_2)
        # l2_init_prob = round(random.uniform(0, 6AGA_O_2), 6AGA_O_2)
        # f1_init_prob = round(random.uniform(0, 6AGA_O_2), 6AGA_O_2)
        # f2_init_prob = round(random.uniform(0, 6AGA_O_2), 6AGA_O_2)

        # # Normalising Probabilities

        # sum_prob = l1_init_prob + l2_init_prob + f1_init_prob + f2_init_prob
        # if sum_prob != 0:
        #     l1_init_prob = round(l1_init_prob / sum_prob,2)
        #     l2_init_prob = round(l2_init_prob / sum_prob,2)
        #     f1_init_prob = round(f1_init_prob / sum_prob,2)
        #     f2_init_prob = round(f2_init_prob / sum_prob,2)

        # diff = 6AGA_O_2 - (l1_init_prob + l2_init_prob + f1_init_prob + f2_init_prob)

        # f2_init_prob += diff

        ## Uniform initial belief means that every type has the same initial probabilty
        l1_init_prob = 0.25
        l2_init_prob = 0.25
        f1_init_prob = 0.25
        f2_init_prob = 0.25

        
        self.l1_estimation.add_estimation_history(round(l1_init_prob, 2),
                                                  round(random.uniform(level_min, level_max), 2),  # 'level'
                                                  round(random.uniform(radius_min, radius_max), 2),  # 'radius'
                                                  round(random.uniform(angle_min, angle_max), 2))  # 'angle'

        self.l2_estimation.add_estimation_history(round(l2_init_prob, 2),
                                                  round(random.uniform(level_min, level_max), 2),  # 'level'
                                                  round(random.uniform(radius_min, radius_max), 2),  # 'radius'
                                                  round(random.uniform(angle_min, angle_max), 2))  # 'angle'

        self.f1_estimation.add_estimation_history(round(f1_init_prob, 2),
                                                  round(random.uniform(level_min, level_max), 2),  # 'level'
                                                  round(random.uniform(radius_min, radius_max), 2),  # 'radius'
                                                  round(random.uniform(angle_min, angle_max), 2))  # 'angle'

        self.f2_estimation.add_estimation_history(round(f2_init_prob, 2),
                                                  round(random.uniform(level_min, level_max), 2),  # 'level'
                                                  round(random.uniform(radius_min, radius_max), 2),  # 'radius'
                                                  round(random.uniform(angle_min, angle_max), 2))  # 'angle'

    ####################################################################################################################
    def get_sampled_probability(self):

        type_probes = list()
        type_probes.append(self.l1_estimation.get_last_type_probability())
        type_probes.append(self.l2_estimation.get_last_type_probability())
        type_probes.append(self.f1_estimation.get_last_type_probability())
        type_probes.append(self.f2_estimation.get_last_type_probability())

        selected_type = choice(types, p=type_probes)  # random sampling the action

        return selected_type

    ####################################################################################################################
    def get_highest_probability(self):

        highest_probability = -1
        selected_type = ''

        for type in types:
            if type == 'l1':
                tmp_prob = self.l1_estimation.get_last_type_probability()

            if type == 'l2':
                tmp_prob = self.l2_estimation.get_last_type_probability()

            if type == 'f1':
                tmp_prob = self.f1_estimation.get_last_type_probability()

            if type == 'f2':
                tmp_prob = self.f2_estimation.get_last_type_probability()

            if tmp_prob > highest_probability:
                highest_probability = tmp_prob
                selected_type = type

        return selected_type

    ####################################################################################################################
    def get_parameters_for_selected_type(self, selected_type):

        if selected_type == 'l1':
            return self.l1_estimation.get_last_estimation()

        if selected_type == 'l2':
            return self.l2_estimation.get_last_estimation()

        if selected_type == 'f1':
            return self.f1_estimation.get_last_estimation()

        if selected_type == 'f2':
            return self.f2_estimation.get_last_estimation()

    ####################################################################################################################
    def update_internal_state(self,main_sim):
        history_index = 0

        for agent in main_sim.agents:
            agent_index = agent.index
            tmp_sim = agent.state_history[history_index]
            tmp_agent = tmp_sim.agents[agent_index]

            for type in types:

                # update internal state for type l1
                tmp_agent.agent_type = type

                if type == 'l1':
                    last_estimation = agent.estimated_parameter.l1_estimation.get_last_estimation()

                if type == 'l2':
                    last_estimation = agent.estimated_parameter.l2_estimation.get_last_estimation()

                if type == 'f1':
                    last_estimation = agent.estimated_parameter.f1_estimation.get_last_estimation()

                if type == 'f2':
                    last_estimation = agent.estimated_parameter.f2_estimation.get_last_estimation()

                self.iteration += 1

                # set latest estimated values to the agent
                tmp_agent.level = last_estimation.level
                tmp_agent.radius = last_estimation.radius
                tmp_agent.angle = last_estimation.angle

                # find the target with
                tmp_agent.visible_agents_items(tmp_sim.items, tmp_sim.agents)
                target = tmp_agent.choose_target(tmp_sim.items, tmp_sim.agents)

                if type == 'l1':
                    last_estimation = agent.estimated_parameter.l1_estimation.internal_state = target

                if type == 'l2':
                    last_estimation = agent.estimated_parameter.l2_estimation.internal_state = target

                if type == 'f1':
                    last_estimation = agent.estimated_parameter.f1_estimation.internal_state = target

                if type == 'f2':
                    last_estimation = agent.estimated_parameter.f2_estimation.internal_state = target

    ####################################################################################################################
    # =================Generating  D = (p,f(p)) , f(p) = P(a|H_t_1,teta,p)==============================================
    def generate_data_for_update_parameter(self, sim, tmp_agent,  action):

        # print '*********************************************************************************'
        # print '******generating data for updating parameter *******'

        D = []  # D= (p,f(p)) , f(p) = P(a|H_t_1,teta,p)

        for i in range(0, self.generated_data_number):

            # Generating random values for parameters
            # tmp_radius = (round(random.uniform(radius_min, radius_max), 2))  # 'radius'
            # tmp_angle = (round(random.uniform(angle_min, angle_max), 2))  # 'angle'
            # tmp_level = (round(random.uniform(level_min, level_max), 2))  # 'level'

            tmp_radius = radius_min + (1.0 * (radius_max - radius_min) / self.generated_data_number) * i
            tmp_angle = angle_min + (1.0 * (angle_max - angle_min) / self.generated_data_number) * i
            tmp_level = level_min + (1.0 * (level_max - level_min) / self.generated_data_number) * i

            tmp_agent.set_parameters(sim, tmp_level, tmp_radius, tmp_angle)

            tmp_agent = sim.move_a_agent(tmp_agent, True)  # f(p)
            p_action = tmp_agent.get_action_probability(action)

            if p_action is not None:
                D.append([tmp_level,tmp_radius, tmp_angle,  p_action])

        return D

    ####################################################################################################################
    def set_choose_target_state(self,state, agent_type):
        if agent_type == 'l1':

            self.l1_estimation.choose_target_state = state

        if agent_type == 'l2':

            self.l2_estimation.choose_target_state = state

        if agent_type == 'f1':

             self.f1_estimation.choose_target_state = state

        if agent_type == 'f2':

            self.f2_estimation.choose_target_state = state

    ####################################################################################################################
    def choosing_false_parameters(self,false_data_set,choose_target_state,new_action,data_set,time_step):
        remove_pf = []
        if time_step > 0:
            if choose_target_state is not None:
                for i in range(len(data_set)):
                    particle_filter = data_set[i]
                    # print particle_filter['route']
                    # print self.action_step_num
                    # print new_action
                    if len(particle_filter['route']) > self.action_step_num :
                        if particle_filter['route'][self.action_step_num] != new_action:
                            false_data_set.append(particle_filter['parameter'])
                            remove_pf.append(particle_filter)
        return remove_pf

    # ####################################################################################################################
    def generate_data(self,  cur_sim,  time_step, new_action, cur_agent):
        # print 'direction in estimation', tmp_agent.direction

        data_set = list()
        weight = list()
        false_data_set = list()

        if cur_agent.agent_type == 'l1':
            data_set = self.l1_estimation.data_set
            false_data_set = self.l1_estimation.false_data_set
            choose_target_state = self.l1_estimation.choose_target_state

        if cur_agent.agent_type == 'l2':
            data_set = self.l2_estimation.data_set
            false_data_set = self.l2_estimation.false_data_set
            choose_target_state = self.l2_estimation.choose_target_state

        if cur_agent.agent_type == 'f1':
            data_set = self.f1_estimation.data_set
            false_data_set = self.f1_estimation.false_data_set
            choose_target_state = self.f1_estimation.choose_target_state

        if cur_agent.agent_type == 'f2':
            data_set = self.f2_estimation.data_set
            false_data_set = self.f2_estimation.false_data_set
            choose_target_state = self.f2_estimation.choose_target_state

        remove_pf = []
        if new_action != 'L':
            remove_pf = self.choosing_false_parameters(false_data_set, choose_target_state, new_action, data_set, time_step)

        actions_history = cur_agent.actions_history
        load_info = {}
        if time_step == 0:
            choose_target_state = deepcopy(cur_sim)
            self.set_choose_target_state(cur_sim ,cur_agent.agent_type)
            load_info['choose_target_state'] = choose_target_state
            load_info['loaded_item'] = None
            self.load_states.append(load_info)
        else:
            if new_action == 'L':
                self.set_choose_target_state(None,cur_agent.agent_type)

                self.action_step_num = 0

            # old_action = actions_history[time_step - 6AGA_O_2]
            if choose_target_state is None and len(actions_history) > 0:
                choose_target_state = deepcopy(cur_sim)
                self.load_states[-1]['loaded_item'] = deepcopy(cur_sim.agents[0].last_loaded_item)
                load_info['choose_target_state'] = choose_target_state
                load_info['loaded_item'] = None
                self.load_states.append(load_info)
                self.set_choose_target_state(cur_sim, cur_agent.agent_type)
                for d in data_set:
                    tmp_agent = (choose_target_state.agents[0])
                    tmp_agent.reset_memory()

                    parameters = d['parameter']
                    tmp_agent.set_parameters(choose_target_state, parameters[0], parameters[1], parameters[2])
                    tmp_agent = choose_target_state.move_a_agent(tmp_agent, True)  # f(p)
                    target = tmp_agent.get_memory()
                    p_action = tmp_agent.get_action_probability(actions_history[0])
                    route_actions = tmp_agent.route_actions

                    if route_actions is not None:
                        if p_action > self.PF_add_threshold and route_actions[
                                                                0:len(actions_history)] == actions_history:
                            w = self.PF_weight
                            d['target'] = target
                            d['weight'] = w
                            d['action_probability'] = p_action
                            d['route'] = tmp_agent.route_actions
                        else:
                            w = 0  # 6AGA_O_2 / self.PF_weight
                            false_data_set.append(d['parameter'])
                            remove_pf.append(d)

        for d in remove_pf:
            data_set.remove(d)
        if len(actions_history) > 0:
            j = 0
            #while j < (self.generated_data_number - len(data_set)):
            for i in range (self.generated_data_number - len(data_set)):
                particle_filter = {}
                tmp_agent = (choose_target_state.agents[0])
                tmp_agent.reset_memory()
                # Generating random values for parameters
                tmp_radius = random.uniform(radius_min, radius_max)  # 'radius'
                tmp_angle = random.uniform(angle_min, angle_max)    # 'angle'
                tmp_level = random.uniform(level_min, level_max)  # 'level'

                if [tmp_level, tmp_radius, tmp_angle] not in false_data_set:

                    tmp_agent.set_parameters(choose_target_state, tmp_level, tmp_radius, tmp_angle)

                    tmp_agent = choose_target_state.move_a_agent(tmp_agent, True)  # f(p)
                    target = tmp_agent.get_memory()

                    p_action = tmp_agent.get_action_probability(actions_history[0])
                    route_actions = tmp_agent.route_actions

                    if route_actions is not None:
                        if p_action > self.PF_add_threshold and route_actions[0:len(actions_history)] == actions_history:
                            w = self.PF_weight
                        else:
                            w = 0 #6AGA_O_2 / self.PF_weight

                        particle_filter['target'] = target

                        particle_filter['parameter'] = [tmp_level, tmp_radius, tmp_angle]
                        # particle_filter['weight'] = w
                        # particle_filter['load_state'] = choose_target_state
                        # particle_filter['action_probability'] = p_action
                        particle_filter['route'] = tmp_agent.route_actions

                        # print(target)
                        # print [tmp_level, tmp_radius, tmp_angle]
                        # print(p_action)

                        if w > 0 and self.check_previous_target(tmp_agent, data_set):
                        #if w > 0:
                            data_set.append(particle_filter)
                            j +=1



        # print(data_set)

        return

    ####################################################################################################################
    def check_previous_target(self,data_set,tmp_agent):
        if_add = True
        remove_parameters = []
        for l in self.load_states:
            previous_load_state = l['choose_target_state']
            if l['loaded_item'] is not None:
                loaded_item_pos = l['loaded_item'].get_position()
                for data in data_set:
                    par = data['parameter']
                    tmp_agent.set_parameters(previous_load_state, par[0], par[1], par[2])
                    tmp_agent = previous_load_state.move_a_agent(tmp_agent, True)  # f(p)
                    target = tmp_agent.get_memory()
                    if loaded_item_pos != target:
                        remove_parameters.append(data)
                        return False
        return True

    ####################################################################################################################

    def get_parameter(self, parameter, index):
        #TODO: Level = 0, angle = 6AGA_O_2, radius = 2? Perhaps there should be a nicer way to do this

        if (index == 0):
            return parameter.level
        if (index == 1):
            return parameter.angle
        if (index == 2):
            return parameter.radius
    
    ####################################################################################################################
    def calculate_gradient_ascent(self,x_train, y_train, old_parameter, polynomial_degree=2, univariate=True):
        # p is parameter estimation value at time step t-6AGA_O_2 and D is pair of (p,f(p))
        # f(p) is the probability of action which is taken by unknown agent with true parameters at time step t
        # (implementation of Algorithm 2 in the paper for updating parameter value)

        step_size = 0.05
        
        if (not univariate):

            reg = linear_model.LinearRegression()

            reg.fit(x_train, y_train)

            gradient = reg.coef_

            # f_coefficients = np.polynomial.polynomial.polyfit(x_train, y_train,
            #                                                   deg=self.polynomial_degree, full=False)

            new_parameters = old_parameter.update(gradient * step_size)

            ## Not sure if we need this rounding
            #new_parameters.level, new_parameters.angle, new_parameters.radius = \
            #    round(new_parameters.level, 2), round(new_parameters.angle, 2), round(new_parameters.radius, 2)

            if new_parameters.level < level_min:
                new_parameters.level = level_min

            if new_parameters.level > level_max:
                new_parameters.level = level_max

            if new_parameters.angle < angle_min:
                new_parameters.angle = angle_min

            if new_parameters.angle > angle_max:
                new_parameters.angle = angle_max

            if new_parameters.radius < radius_min:
                new_parameters.radius = radius_min

            if new_parameters.radius > radius_max:
                new_parameters.radius = radius_max

            return new_parameters
        
        else:

            #import ipdb; ipdb.set_trace()
            
            parameter_estimate = []

            for i in range(len(x_train[0])):

                # Get current independent variables
                current_parameter_set = [elem[i] for elem in x_train]

                # Obtain the parameter in questions upper and lower limits
                p_min = old_parameter.min_max[i][0]
                p_max = old_parameter.min_max[i][1]

                # Fit polynomial to the parameter being modelled
                f_poly = np.polynomial.polynomial.polyfit(current_parameter_set, y_train,
                                                                  deg=polynomial_degree, full=False)

                f_poly = np.polynomial.polynomial.Polynomial(coef=f_poly,domain=[p_min, p_max],window=[p_min, p_max])

                # get gradient
                f_poly_deriv = f_poly.deriv()

                current_estimation = self.get_parameter(old_parameter,i)
                
                delta = f_poly_deriv(current_estimation)

                # update parameter
                new_estimation = current_estimation + step_size*delta

                if (new_estimation < p_min):
                    new_estimation = p_min
                if (new_estimation > p_max):
                    new_estimation = p_max
                
                parameter_estimate.append(new_estimation)

            #print('Parameter Estimate: {}'.format(parameter_estimate))
            
            return Parameter(parameter_estimate[0], parameter_estimate[1], parameter_estimate[2])
        
    ####################################################################################################################
    def calculate_EGO(self, agent_type, time_step):  # Exact Global Optimisation

        multiple_results = 1
        if agent_type.agent_type == 'l1':
            for i in range(0,time_step):
                multiple_results = multiple_results #* self.p_action_parameter_type_l1[i]

        if agent_type.agent_type == 'l2':
            self.p_action_parameter_type_l2 = []

        if agent_type.agent_type == 'f1':
            self.p_action_parameter_type_f1 = []

        if agent_type.agent_type == 'f2':
            self.p_action_parameter_type_f2 = []

        return

    ####################################################################################################################
    def multivariate_bayesian(self, x_train, y_train, previous_estimate):
        # TODO: This method must be called once, not four times in a loop as is currently the case
        np.random.seed(123)

        # Fit multivariate polynomial of degree 4PF_O_2
        f_poly = linear_model.LinearRegression(fit_intercept=True)
        f_poly.fit(x_train, y_train)

        # Extract polynomial coefficients
        f_coefficients = np.insert(f_poly.coef_, 0, f_poly.intercept_)
        logging.info('f-hat Coefficients: {}'.format(f_coefficients))

        # Generate prior
        if self.iteration == 0:
            beliefs = st.uniform.rvs(0, 1, size=4)
            logging.info('Randomly Sampling Beliefs From Standard Uniform')
        else:
            beliefs = previous_estimate.observation_history[-1]

        logging.info('Beliefs at Iteration {}: {}'.format(previous_estimate.iteration, beliefs))

        # Catch array broadcasting errors
        assert len(beliefs) == len(f_coefficients), 'F coefficient and beliefs of differing lengths'
        if len(beliefs) != len(f_coefficients):
            logging.warning('Iteration {}, beliefs and f-hat coefficients of differing lengths.'.format(self.iteration))
            logging.warning('Beliefs Length: {}\nCoefficients Length: {}'.format(len(beliefs), len(f_coefficients)))

        # Compute Improper Posterior Posterior
        g_hat = f_coefficients * beliefs
        logging.info('Polynomial Convolution g-hat values: {}'.format(g_hat))

        # Collect samples from g
        sampled_x = np.linspace(0, 1, 4)
        sampled_y = st.uniform.rvs(0.1, 1, 4)  # TODO: How can I get g(p^l) here?

        # Fit h-hat
        h_polynomial = linear_model.LinearRegression(fit_intercept=True)
        h_polynomial.fit(sampled_x, sampled_y)
        h_coefficients = np.insert(h_polynomial.coef_, 0, h_polynomial.intercept_)

        # Integrate h-hat
        def integrand(level, radius, angle, x):
            pass

        logging.info('Estimation Complete\n{}'.format('-' * 100))

        ####################################################################################################################

    ####################################################################################################################
    def findMin(self,polynomial):
        derivative = polynomial.deriv()

        roots = derivative.roots()

        minValue = sys.maxsize

        for r in roots:
            if (polynomial(r) < minValue):
                minValue = polynomial(r)

        if (polynomial(polynomial.domain[0]) < minValue):
            minValue = polynomial(polynomial.domain[0])

        if (polynomial(polynomial.domain[1]) < minValue):
            minValue = polynomial(polynomial.domain[1])

        return minValue

    def inversePolynomial(self,polynomialInput, y):
        solutions = list()

        polynomial = polynomialInput.copy()
        
        polynomial.coef[0] = polynomial.coef[0] - y

        roots = polynomial.roots()

        for r in roots:
            if (r >= polynomial.domain[0] and r <= polynomial.domain[1]):
                if (not (isinstance(r,complex))):
                    solutions.append(r)
                elif (r.imag == 0):
                    solutions.append(r.real)

        ## We should always have one solution for the inverse?
        if (len(solutions) > 1):
            print "Warning! Multiple solutions when sampling for ABU"
        
        return solutions[0]
    
    ## Inverse transform sampling
    ## https://en.wikipedia.org/wiki/Inverse_transform_sampling
    def sampleFromBelief(self,polynomial,sizeList):
        returnMe = [None]*sizeList

        ## To calculate the CDF, I will first get the integral. The lower part is the lowest possible value for the domain
        ## Given a value x, the CDF will be the integral at x, minus the integral at the lowest possible value.
        dist_integ = polynomial.integ()
        lower_part = dist_integ(polynomial.domain[0])
        cdf = dist_integ.copy()
        cdf.coef[0] = cdf.coef[0] - lower_part
    
        for s in range(sizeList):
            u = np.random.uniform(0,1)

            returnMe[s] = self.inversePolynomial(cdf, u)

        return returnMe
    
    def bayesian_updating(self, x_train, y_train, previous_estimate,  polynomial_degree=2, sampling='average'):

        parameter_estimate = []

        for i in range(len(x_train[0])):
            # Get current independent variables
            current_parameter_set = [elem[i] for elem in x_train]

            # Obtain the parameter in questions upper and lower limits
            p_min = previous_estimate.min_max[i][0]
            p_max = previous_estimate.min_max[i][1]

            # Fit polynomial to the parameter being modelled
            f_poly = np.polynomial.polynomial.polyfit(current_parameter_set, y_train,
                                                              deg=polynomial_degree, full=False)
            
            f_poly = np.polynomial.polynomial.Polynomial(coef=f_poly,domain=[p_min, p_max],window=[p_min, p_max])
            
            # Generate prior
            if self.iteration == 0:
                #beliefs = st.uniform.rvs(0, 6AGA_O_2, size=polynomial_degree + 6AGA_O_2)
                beliefs = [0]*(polynomial_degree + 1)
                beliefs[0] = 1.0/(p_max - p_min)
                
                current_belief_poly = np.polynomial.polynomial.Polynomial(coef=beliefs,domain=[p_min, p_max],window=[p_min,p_max])
            else:
                current_belief_poly = self.belief_poly[i]
            

            # Compute convolution
            g_poly = current_belief_poly*f_poly

            # Collect samples
            # Number of evenly spaced points to compute polynomial at
            # TODO: Not sure why it was polynomial_degree + 6AGA_O_2
            # spacing = polynomial_degree + 6AGA_O_2
            spacing = len(x_train)

            # Generate equally spaced points, unique to the parameter being modelled
            X = np.linspace(p_min, p_max, spacing)
            y = np.array([g_poly(j) for j in X])

            # Future polynomials are modelled using X and y, not D as it's simpler this way. I've left D in for now
            # TODO: possilby remove D if not needed at the end
            D = [(X[j], y[j]) for j in range(len(X))]

            # Fit h
            h_hat_coefficients = np.polynomial.polynomial.polyfit(X, y, deg=polynomial_degree, full=False)
            
            h_poly = np.polynomial.polynomial.Polynomial(coef=h_hat_coefficients,domain=[p_min, p_max],window=[p_min, p_max])

            # "Lift" the polynomial. Perhaps this technique is different than the one in Albrecht and Stone 2017.
            min_h = self.findMin(h_poly)
            if (min_h < 0):
                h_poly.coef[0] = h_poly.coef[0] - min_h

            # Integrate h
            integration = h_poly.integ()

            # Compute I
            definite_integral = integration(p_max) - integration(p_min)

            # Update beliefs
            new_belief_coef = np.divide(h_poly.coef, definite_integral)  # returns an array
            new_belief = np.polynomial.polynomial.Polynomial(coef=new_belief_coef,domain=[p_min, p_max],window=[p_min, p_max])

            self.belief_poly[i] = new_belief

            # TODO: Not better to derivate and get the roots?
            if sampling == 'MAP':
                # Sample from beliefs
                polynomial_max = 0
                granularity = 1000
                x_vals = np.linspace(p_min, p_max, granularity)
                for j in range(len(x_vals)):
                    proposal = new_belief(x_vals[j])
                    print('Proposal: {}'.format(proposal))
                    if proposal > polynomial_max:
                        polynomial_max = proposal

                parameter_estimate.append(polynomial_max)

            elif sampling == 'average':
                x_random = self.sampleFromBelief(new_belief, 10)
                parameter_estimate.append(np.mean(x_random))

            # Increment iterator

        new_parameter = Parameter(parameter_estimate[0], parameter_estimate[1], parameter_estimate[2])
        print('Parameter Estimate: {}'.format(parameter_estimate))
        self.iteration += 1

        return new_parameter

    ####################################################################################################################
    def parameter_estimation(self,time_step, cur_agent, current_sim, action):

        estimated_parameter = None

        last_parameters_value = 0

        if cur_agent.agent_type == 'l1':
            last_parameters_value = deepcopy(self.l1_estimation.get_last_estimation())

        if cur_agent.agent_type == 'l2':
            last_parameters_value = deepcopy(self.l2_estimation.get_last_estimation())

        if cur_agent.agent_type == 'f1':
            last_parameters_value = deepcopy(self.f1_estimation.get_last_estimation())

        if cur_agent.agent_type == 'f2':
            last_parameters_value = deepcopy(self.f2_estimation.get_last_estimation())

        if self.parameter_estimation_mode == 'PF':
            self.generate_data(current_sim, time_step, action, cur_agent)
            current_data_set = list()
            # current_weight = list()
            if cur_agent.agent_type == 'l1':
                current_data_set = self.l1_estimation.data_set
            #     current_weight = self.l1_estimation.weight
            #
            if cur_agent.agent_type == 'l2':
                current_data_set = self.l2_estimation.data_set
            #     current_weight = self.l2_estimation.weight
            #
            if cur_agent.agent_type == 'f1':
                current_data_set = self.f1_estimation.data_set
            #     current_weight = self.f1_estimation.weight
            #
            if cur_agent.agent_type == 'f2':
                current_data_set = self.f2_estimation.data_set
            #     current_weight = self.f2_estimation.weight
            #
            # if current_data_set == []:
            #     return None
            #
            # print(current_data_set)
            # print('--------------------------')
            # print(current_weight)
            # print('--------------------------')
            #
            parameters = []
            for ds in current_data_set:
                parameters.append(ds["parameter"])

            # print parameters
            a_data_set = np.transpose(np.array(parameters))

            if a_data_set != []:
                # a_weights = np.array(current_weight)
                #
                levels = a_data_set [0, :]
                ave_level = np.average(levels) #, weights=a_weights)
                #
                angle = a_data_set[1, :]
                ave_angle = np.average(angle) #, weights=a_weights)
                #
                radius = a_data_set[2, :]
                ave_radius = np.average(radius) #, weights=a_weights)
                new_parameter = Parameter(ave_level, ave_angle, ave_radius)
                print 'new_parameter' , ave_level, ave_angle, ave_radius
                # new_parameter = deepcopy(self.l1_estimation.get_last_estimation())

                return new_parameter
            else :
                return last_parameters_value

        else:

            D = self.generate_data_for_update_parameter(current_sim, cur_agent, action)

            x_train = []
            y_train = []

            if len(D) == 0:
                return

            # Extract x, y train from generated data
            for i in range(0, self.generated_data_number):
                x_train.append(D[i][0:3])
                y_train.append(D[i][3])


            # D = (p,f(p)) , f(p) = P(a|H_t_1,teta,p)
            if self.parameter_estimation_mode == 'AGA':
                estimated_parameter = self.calculate_gradient_ascent(x_train, y_train, last_parameters_value)

            if self.parameter_estimation_mode == 'ABU':
                estimated_parameter = self.bayesian_updating(x_train, y_train, last_parameters_value)

        return estimated_parameter

    ####################################################################################################################
    def nested_list_sum(self, nested_lists):
        if type(nested_lists) == list:
            return np.sum(self.nested_list_sum(sublist) for sublist in nested_lists)
        else:
            return 1

    ####################################################################################################################
    def UCB_selection(self, time_step, final=False):
        agent_types = ['l1', 'l2', 'f1', 'f2']

        # Get the total number of probabilities
        prob_count = self.nested_list_sum(agent_types)

        # Return the mean probability for each type of bandit
        mean_probabilities = [np.mean(i) for i in agent_types]

        # Confidence intervals from standard UCB formula
        cis = [np.sqrt((2 * np.log(prob_count)) / len(agent_types[i]) + 0.01) for i in range(len(agent_types))]

        # Sum together means and CIs
        ucb_values = np.array(mean_probabilities) + np.array(cis)

        # Get max UCB value
        max_index = np.argmax(ucb_values)

        # Determine Agent Type to return
        try:
            if max_index == 0:
                return_agent = ['l1']
            elif max_index == 1:
                return_agent = ['l2']
            elif max_index == 2:
                return_agent = ['f1']
            elif max_index == 3:
                return_agent = ['f2']
            else:
                print('UCB has not worked correctly, defaulting to l1')
                return_agent = ['l1']
        except:
            print('An error has occured in UCB, resorting to l1')
            return_agent = ['f1']

        print('UCB Algorithm returned agent of type: {}'.format(return_agent[0]))

        if final:
            return return_agent
        else:
            return ['f2']

        # nu = 0.1AGA_O_2
        # n = 10
        # parameter_diff_sum =0
        # for i in range (3):
        #     parameter_diff_sum += abs(self.parameters_values_l1[i] - self.parameters_values_l1 [i-6AGA_O_2])
        # reward = (6AGA_O_2/nu) * parameter_diff_sum
        # return ['l1']

    ###################################################################################################################
    def process_parameter_estimations(self, time_step,  agent_direction, action, agent_index,
                                      actions_history, previous_state):

        new_parameters_estimation = None
        selected_types = None

        # tmp_sim = deepcopy(state_history[time_step]) # current state
        tmp_sim = previous_state
        self.sim = tmp_sim
        (x, y) = tmp_sim.agents[agent_index].get_position()  # Position in the world e.g. 2,3
        self.estimated_agent = agent.Agent(x, y, agent_direction, None, agent_index)

        # Start parameter estimation
        if self.type_selection_mode == 'AS':
            selected_types = types
        if self.type_selection_mode == 'BS':
            selected_types = self.UCB_selection(time_step)  # returns l1, l2, f1, f2

        selected_types = ['l1']
        # Estimate the parameters
        for selected_type in selected_types:
            # Generates an Agent object
            tmp_agent = deepcopy(self.estimated_agent)
            tmp_agent.agent_type = selected_type
            tmp_agent.actions_history = actions_history
            tmp_agent.previous_state = previous_state

            # Return new parameters, applying formulae stated in paper Section 1AGA_O_2.2 - list of length 3
            new_parameters_estimation = self.parameter_estimation(time_step, tmp_agent, tmp_sim, action)

            if new_parameters_estimation is not None:
                # moving temp agent in previous map with new parameters
                tmp_agent.set_parameters(tmp_sim, new_parameters_estimation.level, new_parameters_estimation.radius,
                                         new_parameters_estimation.angle)

                # Runs a simulator object
                tmp_agent = tmp_sim.move_a_agent(tmp_agent)

                action_prob = tmp_agent.get_action_probability(action)
                print 'action' , action, 'action_prob' , action_prob

                # TODO: CHANGE IT
                # if time_step > 0:
                #     self.update_internal_state(tmp_sim)

                # Determine which list to append new parameter estimation and action prob to
                if selected_type == 'l1':
                    self.l1_estimation.update_estimation(new_parameters_estimation, action_prob)
                    self.l1_estimation.type_probability = action_prob * self.l1_estimation.get_last_type_probability()

                if selected_type == 'l2':
                    self.l2_estimation.update_estimation(new_parameters_estimation, action_prob)
                    self.l2_estimation.type_probability = action_prob * self.l2_estimation.get_last_type_probability()

                if selected_type == 'f1':
                    self.f1_estimation.update_estimation(new_parameters_estimation, action_prob)
                    self.f1_estimation.type_probability = action_prob * self.f1_estimation.get_last_type_probability()

                if selected_type == 'f2':
                    self.f2_estimation.update_estimation(new_parameters_estimation, action_prob)
                    self.f2_estimation.type_probability = action_prob * self.f2_estimation.get_last_type_probability()

        self.normalize_type_probabilities()
        self.action_step_num += 1

        return new_parameters_estimation

    def normalize_type_probabilities(self):

        l1_update_belief_value = self.l1_estimation.type_probability
        l2_update_belief_value = self.l2_estimation.type_probability
        f1_update_belief_value = self.f1_estimation.type_probability
        f2_update_belief_value = self.f2_estimation.type_probability

        sum_of_probabilities = l1_update_belief_value + l2_update_belief_value + f1_update_belief_value + f2_update_belief_value

        belief_factor = 1

        if sum_of_probabilities != 0:
            belief_factor = 1 / sum_of_probabilities

        l1_prob = l1_update_belief_value * belief_factor
        l2_prob = l2_update_belief_value * belief_factor
        f1_prob = f1_update_belief_value * belief_factor
        f2_prob = f2_update_belief_value * belief_factor

        self.l1_estimation.type_probabilities.append( l1_prob)
        self.l2_estimation.type_probabilities.append( l2_prob)
        self.f1_estimation.type_probabilities.append( f1_prob)
        self.f2_estimation.type_probabilities.append( f2_prob)


