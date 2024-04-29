# RL for mobile-env
This github repository is part of the final project requirement for CS394R. In this project, we focus on developing Reinforcement Learning algorithms to optimize the overall capacity of a telecommunication system by leveraging the [mobile-env](https://github.com/stefanbschneider/mobile-env) Farama Gymnasium environment. mobile-env is a Gymnasium environment which replicate the telecommunication problem where a dynamic mobile-user (UE) is connecting to different basestations (BS) as it moves. We hope to leverage mobile-env to train and test RL algorithms which determines which BS a UE should connect to. Although our project briefly tests the the centralized problem, where knowledge of all connections between UEs and BSs is known by a central agent, the decentralized problem, where each UE acts based on their own interest, is the main focus.

## Prerequisites
To ensure that all necessary packages are downloaded to run this repository, please run the following command:
```
pip install -r requirements.txt
```

## Running Code
In our current demo environment, the code runs a small multi-agent scenario where each UE is determining whether it should connect to a BS, and which one if it decides to connect, independently. In the small environment, there are three BSs and 5 UEs. When executing a run, [main.py](main.py) can take various arguments, which are listed below.

1. `--agent` determines the policy being used, which has options `['reinforce','utility','snr']`. `reinforce` employs the REINFORCE algorithm, whereas `utility` and `SNR` are greedy based policies using utility or SNR as the metric. The default value is `utility`.

2. `--baseline` is of type `bool` and it sets whether the REINFORCE algorithm will have a baseline. The default value is `TRUE`.

3. `--episodes` is of type `int` and it sets the number of episodes you'd like to run. The default value is `10000`.

4. `--name` is of type `str` and it sets the name of the run. The default name is `Test`.

5. `--num_iter` is of type `int` and it sets the number of times you'd like to run the algorithm. The default value is `1`

6. `--log` is of type `int` and it determines whether you'd like to log the runs' metrics via wandb. The default value is `1`.

### Example
```
python3 main.py --agent reinforce --baseline FALSE
```

