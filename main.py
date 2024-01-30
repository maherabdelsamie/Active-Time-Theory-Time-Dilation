import numpy as np
import random
import matplotlib.pyplot as plt

# Constants
C = 299792458  # Speed of light in m/s

# GlobalTime Class Definition
class GlobalTime:
    def __init__(self):
        self.current_time = 0
        self.time_flow_rate = 1
        self.in_generative_phase = False

    def enter_generative_phase(self):
        self.in_generative_phase = True

    def exit_generative_phase(self):
        self.in_generative_phase = False

    def generative_fluctuation(self):
        if self.in_generative_phase:
            fluctuation = self._calculate_fluctuation()
            self.current_time += fluctuation

    def _calculate_fluctuation(self):
        if not self.in_generative_phase:
            return 0

        distributions = {
            "normal": {"mean": 0, "std_dev": 0.02},
            "lognormal": {"mean": 0, "sigma": 0.5},
            "uniform": {"low": -0.05, "high": 0.05}
        }

        distribution_name = np.random.choice(list(distributions.keys()))
        params = distributions[distribution_name]

        if distribution_name == "normal":
            return np.random.normal(params["mean"], params["std_dev"])
        elif distribution_name == "lognormal":
            return np.random.lognormal(params["mean"], params["sigma"])
        elif distribution_name == "uniform":
            return np.random.uniform(params["low"], params["high"])

    def directive_influence(self, particles):
        avg_state = sum(p.state for p in particles) / len(particles)
        k = 0.1
        for particle in particles:
            influence = k * (avg_state - particle.state)
            particle.apply_directive(influence)

    def adaptive_term(self, particles):
        # Example implementation - adapt based on system state
        avg_energy = sum(np.linalg.norm(p.state) ** 2 for p in particles) / len(particles)
        return 0.01 * avg_energy  # Adjust the coefficient as needed

    def generative_noise(self):
        if self.in_generative_phase:
            return np.random.normal(0, 0.01)  # Standard deviation of noise
        return 0

    def adapt(self, system_state):
        k = 0.05
        max_flow_rate = 1e8  # Adjusted to prevent too rapid escalation
        if isinstance(system_state, np.ndarray):
            system_state = np.mean(system_state)
        new_flow_rate = self.time_flow_rate * (1 + k * system_state)
        self.time_flow_rate = min(new_flow_rate, max_flow_rate)

    def calculate_entropy(self, particles):
        states = np.array([particle.state for particle in particles])
        return np.var(states)

    def calculate_avg_energy(self, particles):
        total_energy = sum(np.linalg.norm(particle.state) ** 2 for particle in particles)
        return total_energy / len(particles)

    def calculate_complexity(self, particles):
        return sum(particle.entropy for particle in particles if particle.entropy is not None)

# QuantumParticle Class Definition
class QuantumParticle:
    def __init__(self, state, velocity):
        self.state = np.array(state, dtype=complex)
        self.velocity = velocity
        self.entropy = None
        # Hamiltonian
        self.H = np.array([[0, 1], [1, 0]], dtype=complex)

    def apply_directive(self, influence, noise_std_dev=0.01):
        noise = np.random.normal(0, noise_std_dev, len(self.state))
        self.state += np.array(influence, dtype=complex) + noise

    def apply_fluctuation(self, fluctuation):
        self.state += fluctuation

    def calculate_observables(self):
        self.entropy = np.linalg.norm(self.state)

    def calculate_lorentz_factor(self):
        velocity_norm = np.linalg.norm(self.velocity)
        return 1 / np.sqrt(1 - (velocity_norm / C) ** 2)

    def calculate_new_state(self, extrinsic_dt):
        dstate_dt = -1j * self.H @ self.state
        state = self.state + dstate_dt * extrinsic_dt
        state /= np.linalg.norm(state)  # Normalize
        return state

    def update_state(self, intrinsic_dt):
        lorentz_factor = self.calculate_lorentz_factor()
        extrinsic_dt = intrinsic_dt * lorentz_factor
        dstate_dt = -1j * self.H @ self.state
        state = self.state + dstate_dt * extrinsic_dt
        state /= np.linalg.norm(state)  # Normalize
        return state

        # Apply state update over extrinsic_dt
        self.state = self.calculate_new_state(extrinsic_dt)

    def calculate_dilated_time(self, intrinsic_dt):
        lorentz_factor = self.calculate_lorentz_factor()
        return intrinsic_dt * lorentz_factor

      # Function to calculate new state
    def calculate_new_state(dt):
    
        return np.zeros(2, dtype=complex)

    def apply_directive(self, influence):
        self.state += influence 


# Function to run the simulation
# Function to run the simulation
def run_simulation(use_active_time, max_iterations=1000):
    global_time = GlobalTime()
    particles = [QuantumParticle([1.0, 0.0], np.random.uniform(-0.1, 0.1, 3) * C) for _ in range(2)]

    entropy_values = []
    complexity_values = []
    dilated_times = []
    intrinsic_time = 0

    for iteration_count in range(max_iterations):
        intrinsic_dt = 1  # Default time step

        # Modify intrinsic_dt based on active time properties
        if use_active_time:
            # Fluctuations and adaptive adjustments
            intrinsic_dt *= global_time.time_flow_rate
            intrinsic_dt *= (1 + global_time.adaptive_term(particles))
            intrinsic_dt += global_time.generative_noise()

            # Generative and directive influences
            if np.random.random() < 0.1:
                global_time.enter_generative_phase()
            else:
                global_time.exit_generative_phase()
            global_time.generative_fluctuation()
            global_time.directive_influence(particles)

        # Accumulate intrinsic time
        intrinsic_time += intrinsic_dt

        for particle in particles:
            particle.apply_fluctuation(global_time._calculate_fluctuation()) if use_active_time else None
            particle.calculate_observables()
            particle.update_state(intrinsic_dt)
            dilated_time = particle.calculate_dilated_time(intrinsic_dt)
            dilated_times.append(dilated_time)

        avg_energy = global_time.calculate_avg_energy(particles)
        entropy = global_time.calculate_entropy(particles)
        complexity = global_time.calculate_complexity(particles)

        entropy_values.append(entropy)
        complexity_values.append(complexity)

        global_time.adapt(avg_energy)

    return {
        "intrinsic_time": intrinsic_time,
        "entropy": entropy_values,
        "complexity": complexity_values,
        "dilated_times": dilated_times
    }



def calculate_average_dilated_time(results):
    return np.mean(results['dilated_times'])

def execute_simulations(max_iterations=1000):
    results_with_active_time = run_simulation(use_active_time=True, max_iterations=max_iterations)

    avg_dilated_time_with_active_time = calculate_average_dilated_time(results_with_active_time)

    return results_with_active_time['intrinsic_time'], avg_dilated_time_with_active_time

def plot_active_time_comparison(max_iterations=1000):
    # Run simulations and get results for active time scenario
    intrinsic_time_with, avg_dilated_time_with = execute_simulations(max_iterations)

    # Data for plotting
    categories = ['Intrinsic Time', 'Average Dilated Time']
    values = [intrinsic_time_with, avg_dilated_time_with]

    # Create a bar plot for active time properties
    plt.figure(figsize=(10, 6))
    plt.bar(categories, values, color=['blue', 'green'])
    plt.yscale('log')  # Logarithmic scale due to large differences in values
    plt.title('Active Time Properties: Intrinsic vs. Average Dilated Time')
    plt.ylabel('Time')
    plt.tight_layout()
    plt.show()



# Function to plot comparison
def plot_comparison(results_with_time, results_without_time):
    plt.figure(figsize=(12, 6))

    # Plot entropy comparison
    plt.subplot(1, 2, 1)
    plt.plot(results_with_time['entropy'], label='With Active Time')
    plt.plot(results_without_time['entropy'], label='Without Active Time')
    plt.xlabel('Iteration')
    plt.ylabel('Entropy')
    plt.title('Entropy Comparison')
    plt.legend()

    # Plot complexity comparison
    plt.subplot(1, 2, 2)
    plt.plot(results_with_time['complexity'], label='With Active Time')
    plt.plot(results_without_time['complexity'], label='Without Active Time')
    plt.xlabel('Iteration')
    plt.ylabel('Complexity')
    plt.title('Complexity Comparison')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot time dilation comparison
    plt.figure(figsize=(10, 6))
    plt.plot(results_with_time['dilated_times'], label='With Active Time')
    plt.plot(results_without_time['dilated_times'], label='Without Active Time')
    plt.xlabel('Iteration')
    plt.ylabel('Dilated Time')
    plt.title('Time Dilation Comparison')
    plt.legend()
    plt.show()

 

# Main Execution Block
if __name__ == "__main__":
    results_with_time = run_simulation(True)
    results_without_time = run_simulation(False)
    
    plot_comparison(results_with_time, results_without_time)
    plot_active_time_comparison()
    
    # Print intrinsic and average dilated time for both scenarios
    print("With Active Time:")
    print(f"Intrinsic Time: {results_with_time['intrinsic_time']}")
    print(f"Average Dilated Time: {np.mean(results_with_time['dilated_times'])}")

    print("\nWithout Active Time:")
    print(f"Intrinsic Time: {results_without_time['intrinsic_time']}")
    print(f"Average Dilated Time: {np.mean(results_without_time['dilated_times'])}")
 
