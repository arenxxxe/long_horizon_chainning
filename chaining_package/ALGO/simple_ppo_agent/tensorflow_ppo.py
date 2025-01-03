import tensorflow as tf
import numpy as np
import gym

# Environment setup
env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Hyperparameters
gamma = 0.99  # Discount factor
lr_actor = 0.001  # Actor learning rate
lr_critic = 0.001  # Critic learning rate
clip_ratio = 0.2  # PPO clip ratio
epochs = 10  # Number of optimization epochs
batch_size = 64  # Batch size for optimization

# Actor and Critic networks
class ActorCritic(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super(ActorCritic, self).__init__()
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.policy_logits = tf.keras.layers.Dense(action_size)
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.value = tf.keras.layers.Dense(1)

    def call(self, state):
        x = self.dense1(state)
        logits = self.policy_logits(x)
        value = self.value(x)
        return logits, value

# PPO algorithm
def ppo_loss(old_logits, old_values, advantages, states, actions, returns):
    def compute_loss(logits, values, actions, returns):
        actions_onehot = tf.one_hot(actions, action_size, dtype=tf.float32)
        policy = tf.nn.softmax(logits)
        action_probs = tf.reduce_sum(actions_onehot * policy, axis=1)
        old_policy = tf.nn.softmax(old_logits)
        old_action_probs = tf.reduce_sum(actions_onehot * old_policy, axis=1)

        # Policy loss
        ratio = tf.exp(tf.math.log(action_probs + 1e-10) - tf.math.log(old_action_probs + 1e-10))
        clipped_ratio = tf.clip_by_value(ratio, 1 - clip_ratio, 1 + clip_ratio)
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))

        # Value loss
        value_loss = tf.reduce_mean(tf.square(values - returns))

        # Entropy bonus (optional)
        entropy_bonus = tf.reduce_mean(policy * tf.math.log(policy + 1e-10))

        total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy_bonus  # Entropy regularization
        return total_loss

    def get_advantages(returns, values):
        advantages = returns - values
        return (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)

    def train_step(states, actions, returns, old_logits, old_values):
        with tf.GradientTape() as tape:
            logits, values = model(states)
            loss = compute_loss(logits, values, actions, returns)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    advantages = get_advantages(returns, old_values)
    for _ in range(epochs):
        loss = train_step(states, actions, returns, old_logits, old_values)
    return loss
# Initialize actor-critic model and optimizer
model = ActorCritic(state_size, action_size)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_actor)
def main():


    # Main training loop
    max_episodes = 1000
    max_steps_per_episode = 1000

    for episode in range(max_episodes):
        states, actions, rewards, values, returns = [], [], [], [], []
        state = env.reset()
        for step in range(max_steps_per_episode):
            state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            logits, value = model(state)

            # Sample action from the policy distribution
            action = tf.random.categorical(logits, 1)[0, 0].numpy()
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            values.append(value)

            state = next_state

            if done:
                returns_batch = []
                discounted_sum = 0
                for r in rewards[::-1]:
                    discounted_sum = r + gamma * discounted_sum
                    returns_batch.append(discounted_sum)
                returns_batch.reverse()

                states = tf.concat(states, axis=0)
                actions = np.array(actions, dtype=np.int32)
                values = tf.concat(values, axis=0)
                returns_batch = tf.convert_to_tensor(returns_batch)
                old_logits, _ = model(states)

                loss = ppo_loss(old_logits, values, returns_batch - np.array(values), states, actions, returns_batch)
                print(f"Episode: {episode + 1}, Loss: {loss.numpy()}")

                break