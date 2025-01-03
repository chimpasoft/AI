import pybullet as p
import pybullet_data
import time
import numpy as np
import csv
import json
from population import Population
import signal

# Timeout handler for simulations
def timeout_handler(signum, frame):
    raise TimeoutError("Simulation step timeout exceeded.")

signal.signal(signal.SIGALRM, timeout_handler)

# Function to create the arena
def make_arena(arena_size=10, wall_height=1):
    wall_thickness = 0.5
    floor_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size / 2, arena_size / 2, wall_thickness])
    floor_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size / 2, arena_size / 2, wall_thickness], rgbaColor=[1, 1, 0, 1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=floor_collision_shape, baseVisualShapeIndex=floor_visual_shape, basePosition=[0, 0, -wall_thickness])

    wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size / 2, wall_thickness / 2, wall_height / 2])
    wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[arena_size / 2, wall_thickness / 2, wall_height / 2], rgbaColor=[0.7, 0.7, 0.7, 1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[0, arena_size / 2, wall_height / 2])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=wall_collision_shape, baseVisualShapeIndex=wall_visual_shape, basePosition=[0, -arena_size / 2, wall_height / 2])

    side_wall_collision_shape = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[wall_thickness / 2, arena_size / 2, wall_height / 2])
    side_wall_visual_shape = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[wall_thickness / 2, arena_size / 2, wall_height / 2], rgbaColor=[0.7, 0.7, 0.7, 1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=side_wall_collision_shape, baseVisualShapeIndex=side_wall_visual_shape, basePosition=[arena_size / 2, 0, wall_height / 2])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=side_wall_collision_shape, baseVisualShapeIndex=side_wall_visual_shape, basePosition=[-arena_size / 2, 0, wall_height / 2])

# Simulation settings
arena_size = 20
pop_size = 5
gene_count = 5
num_generations = 50
total_frames_training = 150

# Phase 1: Accelerated Training
p.connect(p.DIRECT)  # Headless mode for accelerated simulation
population = Population(pop_size=pop_size, gene_count=gene_count)

best_creature_dna = None
best_fitness = 0

with open("fitness_results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Generation", "Creature", "Fitness"])

    for generation in range(num_generations):
        print(f"\n--- Generation {generation + 1} ---")
        generation_scores = []

        for i, random_creature in enumerate(population.creatures):
            pos = [
                np.random.uniform(-arena_size / 2, arena_size / 2), 
                np.random.uniform(-arena_size / 2, arena_size / 2), 
                1.5
            ]

            try:
                urdf_file = f"creature_{i}.urdf"
                with open(urdf_file, "w") as f:
                    f.write(random_creature.to_xml())
                creature_id = p.loadURDF(urdf_file, pos, [0, 0, 0, 1])
            except Exception as e:
                print(f"Error loading creature {i}: {e}")
                continue

            for frame in range(total_frames_training):
                try:
                    signal.alarm(2)  # Timeout after 2 seconds per step
                    p.stepSimulation()
                    signal.alarm(0)  # Disable alarm

                    # Apply motor controls
                    if frame % 12 == 0:
                        motors = random_creature.get_motors()
                        for jid in range(p.getNumJoints(creature_id)):
                            if jid < len(motors):
                                motor = motors[jid]
                                p.setJointMotorControl2(
                                    bodyUniqueId=creature_id,
                                    jointIndex=jid,
                                    controlMode=p.VELOCITY_CONTROL,
                                    targetVelocity=motor.get_output(),
                                    force=1000,  # Increased force for faster movement
                                )

                    pos, _ = p.getBasePositionAndOrientation(creature_id)
                    random_creature.update_position(pos)
                except TimeoutError:
                    print(f"Simulation step for creature {i} in generation {generation} timed out.")
                    break

            fitness = random_creature.get_fitness()
            generation_scores.append(fitness)
            writer.writerow([generation + 1, i + 1, fitness])

            if fitness > best_fitness:
                best_fitness = fitness
                best_creature_dna = random_creature.dna

        print(f"Best fitness in Generation {generation + 1}: {max(generation_scores)}")

p.disconnect()

# Save best creature DNA
with open("best_creature_dna.json", "w") as json_file:
    json.dump({"dna": [gene.tolist() for gene in best_creature_dna], "fitness": best_fitness}, json_file)

# Phase 2: Visualize Best Creature
p.connect(p.GUI)  # GUI mode for visualization
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -10)
make_arena(arena_size=arena_size)

# Load mountain
mountain_position = [0, 0, -1]
mountain_orientation = p.getQuaternionFromEuler([0, 0, 0])
mountain = p.loadURDF("shapes/gaussian_pyramid.urdf", mountain_position, mountain_orientation, useFixedBase=1)

# Camera setup
p.resetDebugVisualizerCamera(cameraDistance=30, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0])

# Load and simulate the best-performing creature
print("\nVisualizing the best creature...")
best_creature = population.creatures[0]  # Placeholder to reconstruct
best_creature.update_dna(best_creature_dna)

# Set a spawn position away from the mountain
safe_spawn_position = [4, 4, 1.5]  # Ensure it's far enough from the mountain

urdf_file = "best_creature.urdf"
with open(urdf_file, "w") as f:
    f.write(best_creature.to_xml())
creature_id = p.loadURDF(urdf_file, safe_spawn_position, [0, 0, 0, 1])

total_frames = 2400
for frame in range(total_frames):  # Extended visualization phase
    p.stepSimulation()

    # Apply motor controls
    if frame % 12 == 0:
        motors = best_creature.get_motors()
        for jid in range(p.getNumJoints(creature_id)):
            if jid < len(motors):
                motor = motors[jid]
                p.setJointMotorControl2(
                    bodyUniqueId=creature_id,
                    jointIndex=jid,
                    controlMode=p.VELOCITY_CONTROL,
                    targetVelocity=motor.get_output(),
                    force=1000,
                )

    time.sleep(1 / 240)

p.disconnect()
