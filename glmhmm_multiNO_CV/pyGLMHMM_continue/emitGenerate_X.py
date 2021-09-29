import numpy as np

def _emit_generate(emit_w, trans_w, data):
    # data: X features or stimuli: (regressors, time).
    # emit_w are the weights that we are learning: in format states x weights; states x emissions-1 x filter bins--regressors
    # trans_w are the weights that we are learning: in format states x weights; states x states x filter bins--regressors
    ##
    num_states = trans_w.shape[0]
    num_emissions = emit_w.shape[1]
    num_bins = trans_w.shape[2]
    T = data.shape[1]
    ##output
    output = np.zeros((T)) # y or labels
    output_lik=np.zeros((num_emissions+1,T))##num_emissions=num_emissions_real-1
    state = np.zeros((T+1)) # z or hidden states, omit last one at last
    state_lik=np.zeros((num_states,T+1))
    ##s_guess: state likelihood
    s_guess = np.zeros((num_states))
    for s1 in range(0, num_states):
        s_guess[s1] = 1 / (1 + np.sum(np.exp(np.sum(np.reshape(trans_w[s1, np.setdiff1d(np.arange(0, num_states), s1), :], (num_states - 1, num_bins), order = 'F') * np.tile(data[:, 0].T, (num_states - 1, 1)), axis = 1)), axis = 0))

    state[0] = np.argmax(s_guess)
    state_lik[:,0]=s_guess
    # Whatever, I guess this is just random at this point...

    for t in range(0, T):
        ##calculate output symbols or y
        filtpower = np.exp(np.sum(emit_w[int(state[t]), :, :] * np.tile(np.reshape(data[:, t].T, (1, 1, data.shape[0]), order = 'F'), (1, num_emissions, 1)), axis = 2)).T
        sym_lik=np.concatenate((1/(1 + np.sum(filtpower, axis = 0)),(filtpower / (1 + np.sum(filtpower, axis = 0)))[:,0]))##the first factor is 0 symbol class
        ##
        output_lik[:,t]=sym_lik
        output[t] = np.argmax(sym_lik)
        ##calculate hidden states or z
        filtpower = np.exp(np.sum(np.reshape(trans_w[int(state[t]), np.setdiff1d(np.arange(0, num_states), int(state[t])), :], (num_states - 1, num_bins), order = 'F') * np.tile(data[:, t].T, (num_states - 1, 1)), axis = 1))
        s_lik=filtpower / (1 + np.sum(filtpower, axis = 0))
        s_guess[int(state[t])] = 1 / (1 + np.sum(filtpower, axis = 0))
        ind = 0
        for s1 in np.setdiff1d(np.arange(0, num_states), int(state[t])):
            s_guess[s1] = s_lik[ind]
            ind = ind + 1
        ##store state likelihood
        state_lik[:,t+1]=s_guess
        ##calculate next state
        state[t + 1] = np.argmax(s_guess)

    ##return 0~T-1 state
    state=state[range(T)]
    state_lik=state_lik[:,range(T)]

    return output,output_lik,state,state_lik

def _emit_generate_z0(emit_w, trans_w, data, z0, z0_lik):
    # data: X features or stimuli: (regressors, time).
    # emit_w are the weights that we are learning: in format states x weights; states x emissions-1 x filter bins--regressors
    # trans_w are the weights that we are learning: in format states x weights; states x states x filter bins--regressors
    ##
    num_states = trans_w.shape[0]
    num_emissions = emit_w.shape[1]
    num_bins = trans_w.shape[2]
    T = data.shape[1]
    ##output
    output = np.zeros((T)) # y or labels
    output_lik=np.zeros((num_emissions+1,T))##num_emissions=num_emissions_real-1
    state = np.zeros((T+1)) # z or hidden states, omit last one at last
    state_lik=np.zeros((num_states,T+1))
    ##s_guess: state likelihood
    s_guess = np.zeros((num_states))
    for s1 in range(0, num_states):
        s_guess[s1] = 1 / (1 + np.sum(np.exp(np.sum(np.reshape(trans_w[s1, np.setdiff1d(np.arange(0, num_states), s1), :], (num_states - 1, num_bins), order = 'F') * np.tile(data[:, 0].T, (num_states - 1, 1)), axis = 1)), axis = 0))
    ##initialization states
    if np.isnan(z0):
        state[0] = np.argmax(s_guess)
        state_lik[:,0]=s_guess
    else:
        state[0] = z0
        state_lik[:,0] = z0_lik
        s_guess = z0_lik
    # Whatever, I guess this is just random at this point...

    for t in range(0, T):
        ##calculate output symbols or y
        filtpower = np.exp(np.sum(emit_w[int(state[t]), :, :] * np.tile(np.reshape(data[:, t].T, (1, 1, data.shape[0]), order = 'F'), (1, num_emissions, 1)), axis = 2)).T
        sym_lik=np.concatenate((1/(1 + np.sum(filtpower, axis = 0)),(filtpower / (1 + np.sum(filtpower, axis = 0)))[:,0]))##the first factor is 0 symbol class
        ##
        output_lik[:,t]=sym_lik
        output[t] = np.argmax(sym_lik)
        ##calculate hidden states or z
        filtpower = np.exp(np.sum(np.reshape(trans_w[int(state[t]), np.setdiff1d(np.arange(0, num_states), int(state[t])), :], (num_states - 1, num_bins), order = 'F') * np.tile(data[:, t].T, (num_states - 1, 1)), axis = 1))
        s_lik=filtpower / (1 + np.sum(filtpower, axis = 0))
        s_guess[int(state[t])] = 1 / (1 + np.sum(filtpower, axis = 0))
        ind = 0
        for s1 in np.setdiff1d(np.arange(0, num_states), int(state[t])):
            s_guess[s1] = s_lik[ind]
            ind = ind + 1
        ##store state likelihood
        state_lik[:,t+1]=s_guess
        ##calculate next state
        state[t + 1] = np.argmax(s_guess)

    ##return 0~T-1 state
    state=state[range(T)]
    state_lik=state_lik[:,range(T)]

    return output,output_lik,state,state_lik