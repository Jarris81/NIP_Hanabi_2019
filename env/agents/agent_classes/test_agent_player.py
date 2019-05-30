    ### Set up the environment
    game_type = "Hanabi-Full"
    num_players = 2

    env = xp.create_environment(game_type=game_type, num_players=num_players)

    # Setup Obs Stacker that keeps track of Observation for all agents ! Already includes logic for distinguishing the view between different agents
    history_size = 1
    obs_stacker = xp.create_obs_stacker(env,history_size=history_size)
    observation_size = obs_stacker.observation_size()
    obs_vectorizer = vectorizer.ObservationVectorizer(env)


    ### Set up the RL-Player, reload weights from trained model
    agent = "DQN"

    ### Specify model weights to be loaded
    path = "/home/dg/Projects/RL/Hanabi/NIP_Hanabi_2019/env/agents/experiments/dqn_sp_4pl_1000_it/playable_models"
    iteration_no = 1950
