import pybullet as p
import pybullet_data
import time
import numpy as np
from population import Population  # Import Population class
from simulation import Simulation  # Import Simulation class

# Step 1: Connect to PyBullet
p.connect(p.GUI)  # GUI mode for visualization
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # For accessing PyBullet assets

# Step 2: Function to Create the Arena
def make_arena(arena_size=10, wall_height=1):
    wall_thickness = 0.5
    # Create floor
    floor_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size / 2, arena_size / 2, wall_thickness])
    floor_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size / 2, arena_size / 2, wall_thickness], rgbaColor=[1, 1, 0, 1])  # Yellow floor
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=floor_collision_shape, baseVisualShapeIndex=floor_visual_shape, basePosition=[0, 0, -wall_thickness])

    # Create walls
    wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size / 2, wall_thickness / 2, wall_height / 2])
    wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size / 2, wall_thickness / 2, wall_height / 2], rgbaColor=[0.7, 0.7, 0.7, 1])  # Gray walls
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[0, arena_size / 2, wall_height / 2])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[0, -arena_size / 2, wall_height / 2])
    
    # Side walls
    side_wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[wall_thickness / 2, arena_size / 2, wall_height / 2])
    side_wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[wall_thickness / 2, arena_size / 2, wall_height / 2], rgbaColor=[0.7, 0.7, 0.7, 1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=side_wall_collision_shape, baseVisualShapeIndex=side_wall_visual_shape, basePosition=[arena_size / 2, 0, wall_height / 2])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=side_wall_collision_shape, baseVisualShapeIndex=side_wall_visual_shape, basePosition=[-arena_size / 2, 0, wall_height / 2])

# Step 3: Create the Environment
p.setGravity(0, 0, -10)  # Set gravity
arena_size = 20
make_arena(arena_size=arena_size)  # Initialize arena

# Step 4: Set the Camera
p.resetDebugVisualizerCamera(cameraDistance=25, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])  # Adjust the camera

# Load mountain in the center of the arena
mountain_position = [0, 0, -1]
mountain_orientation = p.getQuaternionFromEuler([0, 0, 0])
p.setAdditionalSearchPath('shapes/')  # Path to custom shapes
mountain = p.loadURDF("gaussian_pyramid.urdf", mountain_position, mountain_orientation, useFixedBase=1)

# Step 5: Create a Population of Creatures
pop_size = 5
gene_count = 3
population = Population(pop_size=pop_size, gene_count=gene_count)

# Step 6: Run Population Simulation
fitness_scores = []

for i, random_creature in enumerate(population.creatures):
    # Save and load the creature's URDF
    urdf_file = f"creature_{i}.urdf"
    with open(urdf_file, "w") as f:
        f.write(random_creature.to_xml())
    creature_id = p.loadURDF(urdf_file, [0, 0, 5.0], [0, 0, 0, 1])

    # Simulation loop
    total_frames = 2400
    for frame in range(total_frames):
        p.stepSimulation()

        if frame % 24 == 0:  # Update motor control every 24 frames
            motors = random_creature.get_motors()
            for jid in range(p.getNumJoints(creature_id)):
                if jid < len(motors):  # Ensure motor count matches joint count
                    motor = motors[jid]
                    p.setJointMotorControl2(
                        bodyUniqueId=creature_id,
                        jointIndex=jid,
                        controlMode=p.VELOCITY_CONTROL,
                        targetVelocity=motor.get_output(),
                        force=10,
                    )

        # Track creature's position
        pos, _ = p.getBasePositionAndOrientation(creature_id)
        random_creature.update_position(pos)

    # Calculate fitness
    distance_travelled = random_creature.get_distance_travelled()
    fitness_scores.append(distance_travelled)
    print(f"Creature {i + 1} traveled distance: {distance_travelled}")

# Print fitness results
print("Fitness scores:", fitness_scores)

# Step 7: Disconnect
p.disconnect()
