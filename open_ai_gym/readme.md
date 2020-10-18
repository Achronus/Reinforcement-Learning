# OpenAI Gym

[OpenAI Gym](https://gym.openai.com) is a Python library called `gym` that provides a rich collection of environments for RL experiments using a unified interface. This readme file outlines and explains many components within this library, allowing users to better understand each component individually.

## Contents

1. [The Environment](#The-Environment)
   - [Step() Method](#Step()-Method)
2. [Observation Space](#Observation-Space)
   - [Space Classes](#Space-Classes)
   - [Container Classes](#Container-Classes)
3. [Wrappers](#Wrappers)
4. [Monitor](#Monitor)
5. [References](#References)

## The Environment

The central class of the library is `Env`, an environment class. Every environment provides the following information and functionality:

- An `action_space` which is a set of actions to be executed in the environment. These can be both discrete or continuous actions.
- An `observation_space` which is the observations shapes and boundaries provided by the environment.
- A `step()` method which allows the agent to take an action and returns the information about the actions outcome.
- A `reset()` method which resets the environment to its initial state and returns the initial observation. This must be called after the creation of the environment.

Additional utility methods can be found in the `Env` class, such as:

- `render()` - Provides the environment in a _human-friendly format_ or _rgb array_.
- `close()` - Closes the environment.
- `seed(seed=None)` - Sets the seed for the environment's random number generator.

### Step() Method

The `step()` method is the central piece in the environment's functionality. Taking in an `action` as a parameter, it carries out four tasks in one call:

  1. Telling the environment which action to execute on the next step.
  2. Getting the new observation from the environment.
  3. Getting the reward the agent gained within that step.
  4. Getting an indication if the episode is over.

Once each task has been completed, a tuple of four components are returned:

- _The next observation_ - a NumPy vector or matrix with observation data.
- _Local reward_ - a float value of the reward.
- _End-of-episode flag_ - a boolean indicator, `True` when the episode is over.
- _Auxiliary diagnostic information_ - anything environment-specific with extra information about the environment. Can be helpful for debugging.

## Observation Space

The observation space is best reflected through the basic abstract class `Space`. The purpose of this class is to define the observation and action spaces, while allowing users to customize environments to fit their requirements. There are two core methods:

- `sample()` - returns a random sample from the space.
- `contains(x)` - checks if the argument `x` belongs to the space's domain.

Additionally, the `Space` class is inherited by other abstract child classes - existing space classes `Box` and `Discrete`; and container classes `Tuple` and `Dict`.

### Space Classes

- `Box` - represents an n-dimensional tensor of rational numbers with intervals [low, high]. For example: `Box(low=0, high=255, shape(210, 160, 3), dtype=np.uint8)` represents an Atari screen observation which is an RGB image of size 210x160. The `shape=(210, 160, 3)` argument is a tuple of three elements: height, width and colour planes (red, green, blue, respectively).

- `Discrete` - represents a mutually-exclusive set of items, numbered from 0 to n-1. Takes in the parameter `n` as a count of items it describes. For example: `Discrete(n=4)` can be used to describe an action space of 4 directions to move [left, right, up, down].

### Container Classes

Allows the combination of multiple `Space` class instances together, providing the opportunity to create action and observation spaces of any complexity.

Using the `Tuple` class as an example: `Tuple(spaces=(Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32), Discrete(n=3), Discrete(n=2)))`.

This could represent an action space specification for a car, where the car has several controls that can be changed at each timestep:

- `Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)` - Steering wheel angle, brake pedal position, accelerator pedal position.
- `Discrete(n=3)` - Turn signal (off, right or left).
- `Discrete(n=2)` - Horn (on, off).

## Wrappers

The `Wrapper` class inherits from the `Env` class. Where its constructor takes only 1 argument: the instance of the `Env` class to be "wrapped". There are 3 types of subclasses of `Wrapper` that allow filtering of specific parts of information.

These are as follows:

- `ObservationWrapper` - the `observation(obs)` method must be overridden where the `obs` argument is an observation from the wrapped environment. This method should return the observation that will be given to the agent.
- `RewardWrapper` - the `reward(rew)` method must be overridden and allows modification of the reward value given to the agent.
- `ActionWrapper` - the `action(act)` method must be overridden and provides the ability to modify the action passed to the wrapped environment to the agent.

To add extra functionality, simply redefine the methods you want to extend. For example: `step()` or `reset()`.

## Monitor

Implemented like the `Wrapper` class, the `Monitor` class can write information about the agent's performance in a file with an optional video recording of the agent in action.

It can be useful to help look at the agent's life inside its environment. The `Monitor` takes in 2 arguments:

1. First - the `environment` variable.
2. Second - the name of the `dictionary` that the `Wrapper` class writes the results to.

This requires the `FFmpeg` utility to be able to function, which is used to convert captured observations into an output video file.

## References

- [OpenAI Gym | GitHub](https://github.com/openai/gym)
- [Maxim Lapan - Deep Reinforcement Learning Hands-On (Second Edition) | Book](https://www.packtpub.com/product/deep-reinforcement-learning-hands-on-second-edition/9781838826994)