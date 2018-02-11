# DeepTraffic

[DeepTraffic](https://selfdrivingcars.mit.edu/deeptraffic/) is a simulation of multi-lane road traffic provided by MIT course [6.S094](https://selfdrivingcars.mit.edu/) during IAP, 2018.

## Features to learn

From my perspective, there should be some features that the network should learn in the training process.

* Speed up when there are spaces in front
* Turn to the lane to the left if there's a car in front and another at right hand side (maybe further), and vice versa.

Some other features that are more expensive to learn includes:

* Predict in advance how to turn without a car in front
* Escape a pitfall by decelerating deliberately.

## Design of the Neural Network

We use a deep q-learning network to evaluate the Q function.  

````javascript
lanesSide = 3;
patchesAhead = 40;
patchesBehind = 5;
````

```javascript
var temporal_window = 0;
```

The input is set have a large width and forward depth so that the car can decide in advance. The temporal window is set to zero as the optimal solution can be decided without knowing about the history.

```javascript
layer_defs.push({
    type: 'fc',
    num_neurons: 216,
    activation: 'relu'
});

layer_defs.push({
    type: 'fc',
    num_neurons: 432,
    drop_prob: 0.25,
    activation: 'relu'
});
```

The body of the network is replace with two dense layers with large number of units. Convolution layer doesn't work as the encoding of the input has lost the spatial relations of unit (See the [operation of game](https://selfdrivingcars.mit.edu/deeptraffic/gameopt.js). A large number neurons have a higher possibility to express different features listed before, especially given the dropout regulation that makes multiple neurons expressing similar features.

```javascript
var tdtrainer_options = {
    learning_rate: 0.001,
    momentum: 0.0,
    batch_size: 256,
    l2_decay: 0.01
};
```

For the optimizer, the batch size is larger to speed up the training process.

## Design of the Reinforcement Learning

```javascript var opt = {};
opt.experience_size = 30000;
opt.gamma = 0.9;
opt.epsilon_min = 0.20;
opt.random_action_distribution=[0.15,0.30,0.05,0.25,0.25]
```

We use larger experience size and smaller weight decade (long term benefits are emphasized). The $\epsilon$ in exploration is set larger to let the network learning other possible options, and the random distribution is set to favor accelerating and tuning over decelerating and holding still. 

## Code

The code is base on a javascript library [ConvNetJS](https://cs.stanford.edu/people/karpathy/convnetjs/)([Github](https://github.com/karpathy/convnetjs)) designed for traing neural networks on CPU.  If you want to run it, just load the javascript file by clicking `Load Code/Net from File` button on the website. 

## Result

The neural network and reinforcement learning strategies that I implemented achieved an averaged speed of 74.05 mph, compared to the speed limit of 80 mph. 

![My Highest Speed](HighestSpeed.png)

The highest averaged speed by Feburary 11, 2018 is 76.04 mph (See [leaderboards](https://selfdrivingcars.mit.edu/deeptraffic-leaderboard/)]), so I assume this implementation performed really well without fine-tuning the network. Also many other competitors are either already working in the industry or studying in grad school, and they must more experience in training neural networks.

## Future Work

The optimal hyperparameters can be learned by grid search, but I currently don't have the computing power to try this method.

Early stopping may also be an option as the network might sometimes misbehave. 

## Acknowledgements

I should sincerely thank [Lex Fridman](http://lexfridman.com/) and his TAs for offering us such an oppotunity to practice deep reinforcement learning in a more pratical scenario. 