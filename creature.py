import genome 
from xml.dom.minidom import getDOMImplementation
from enum import Enum
import numpy as np

class MotorType(Enum):
    PULSE = 1
    SINE = 2

class Motor:
    def __init__(self, control_waveform, control_amp, control_freq):
        if control_waveform <= 0.5:
            self.motor_type = MotorType.PULSE
        else:
            self.motor_type = MotorType.SINE
        self.amp = max(1, control_amp * 100)  # Scale amplitude for stronger motion
        self.freq = max(1, control_freq * 100)  # Scale frequency for rapid motion
        self.phase = 0

    def get_output(self):
        self.phase = (self.phase + self.freq) % (2 * np.pi)
        if self.motor_type == MotorType.PULSE:
            return self.amp if self.phase < np.pi else -self.amp
        return self.amp * np.sin(self.phase)

class Creature:
    def __init__(self, gene_count):
        self.spec = genome.Genome.get_gene_spec()
        self.dna = genome.Genome.get_random_genome(len(self.spec), gene_count)
        self.flat_links = None
        self.exp_links = None
        self.motors = None
        self.start_position = None
        self.last_position = None
        self.max_height = 0  # Track maximum height for fitness
        self.distance_to_peak = float('inf')  # Initialize distance to peak

    def get_flat_links(self):
        if self.flat_links is None:
            gdicts = genome.Genome.get_genome_dicts(self.dna, self.spec)
            self.flat_links = genome.Genome.genome_to_links(gdicts)
        return self.flat_links

    def get_expanded_links(self):
        self.get_flat_links()
        if self.exp_links is not None:
            return self.exp_links
        
        exp_links = [self.flat_links[0]]
        genome.Genome.expandLinks(
            self.flat_links[0], 
            self.flat_links[0].name, 
            self.flat_links, 
            exp_links
        )
        self.exp_links = exp_links
        return self.exp_links

    def to_xml(self):
        self.get_expanded_links()
        domimpl = getDOMImplementation()
        adom = domimpl.createDocument(None, "robot", None)
        robot_tag = adom.documentElement
        
        for link in self.exp_links:
            robot_tag.appendChild(link.to_link_element(adom))
        for link in self.exp_links[1:]:  # Skip the root node
            robot_tag.appendChild(link.to_joint_element(adom))
        robot_tag.setAttribute("name", "creature")  # Assign a name

        return '<?xml version="1.0"?>' + robot_tag.toprettyxml()

    def get_motors(self):
        self.get_expanded_links()
        if self.motors is None:
            self.motors = [
                Motor(l.control_waveform, l.control_amp, l.control_freq)
                for l in self.exp_links[1:]  # Skip root link
            ]
        return self.motors 

    def update_position(self, pos):
        if self.start_position is None:
            self.start_position = pos
        self.last_position = pos
        self.max_height = max(self.max_height, pos[2])
        # Calculate distance to the mountain peak (0, 0, 5)
        mountain_peak = np.array([0, 0, 5])
        self.distance_to_peak = np.linalg.norm(np.array(pos) - mountain_peak)

    def get_distance_travelled(self):
        if self.start_position is None or self.last_position is None:
            return 0
        return np.linalg.norm(np.array(self.start_position) - np.array(self.last_position))

    def get_fitness(self):
        """Calculate fitness based on proximity to the mountain and realistic climbing."""
        if self.last_position is None:
            return 0

        # Extract coordinates
        x, y, z = self.last_position

        # Height reward
        height_reward = z  # Reward for climbing higher

        # Penalize leaving the arena
        arena_size = 20
        if abs(x) > arena_size / 2 or abs(y) > arena_size / 2:
            return 0  # Disqualify creatures leaving the arena

        # Penalize going too far from the mountain
        mountain_base = np.array([0, 0, 0])  # Assuming mountain base is at (0, 0, 0)
        distance_to_mountain = np.linalg.norm(np.array([x, y, z]) - mountain_base)

        if distance_to_mountain > arena_size:
            return 0  # Disqualify creatures going too far from the mountain

        # Combine height reward and proximity to the mountain peak
        mountain_peak = np.array([0, 0, 5])  # Assuming the peak is at (0, 0, 5)
        distance_to_peak = np.linalg.norm(np.array([x, y, z]) - mountain_peak)
        proximity_reward = max(0, 20 - distance_to_peak)  # Reward for getting closer to the peak

        # Final fitness calculation
        fitness = height_reward + proximity_reward - distance_to_peak * 0.1  # Penalize distance to peak
        return max(0, fitness)  # Ensure non-negative fitness


    def update_dna(self, dna):
        self.dna = dna
        self.flat_links = None
        self.exp_links = None
        self.motors = None
        self.start_position = None
        self.last_position = None
        self.max_height = 0  # Reset max height
        self.distance_to_peak = float('inf')  # Reset distance to the peak
