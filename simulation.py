import time
import pybullet as p
from multiprocessing import Pool

class Simulation: 
    def __init__(self, sim_id=0):
        self.physicsClientId = p.connect(p.DIRECT)        
        self.sim_id = sim_id

    def run_creature(self, cr, iterations=2400):
        """
        runs the simulation for a specified creature
        """
        pid = self.physicsClientId
        p.resetSimulation(physicsClientId=pid)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=pid)

        p.setGravity(0, 0, -10, physicsClientId=pid)
        plane_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=pid)
        floor = p.createMultiBody(plane_shape, plane_shape, physicsClientId=pid)

        xml_file = 'temp' + str(self.sim_id) + '.urdf'
        xml_str = cr.to_xml()
        with open(xml_file, 'w') as f:
            f.write(xml_str)
        
        cid = p.loadURDF(xml_file, physicsClientId=pid)

        p.resetBasePositionAndOrientation(cid, [0, 0, 2.5], [0, 0, 0, 1], physicsClientId=pid)


        for step in range(iterations):
            p.stepSimulation(physicsClientId=pid)
            if step % 24 == 0:
                self.update_motors(cid=cid, cr=cr)

            pos, orn = p.getBasePositionAndOrientation(cid, physicsClientId=pid)
            cr.update_position(pos)
            #print(pos[2])
            #print(cr.get_distance_travelled())

    
    def update_motors(self, cid, cr):
        """
        cid is the id in the physics engine
        cr is a creature object
        """
        for jid in range(p.getNumJoints(cid,
                                        physicsClientId=self.physicsClientId)):
            m = cr.get_motors()[jid]

            p.setJointMotorControl2(cid, jid, 
                    controlMode=p.VELOCITY_CONTROL, 
                    targetVelocity=m.get_output(), 
                    force = 5, 
                    physicsClientId=self.physicsClientId)
        

    # You can add this to the Simulation class:
    def eval_population(self, pop, iterations):
        for cr in pop.creatures:
            self.run_creature(cr, 2400) 

    def evaluate_fitness(self, cr, total_frames=2400):
        """
        Evaluates the fitness of a creature based on maximum height achieved.
        """
        pid = self.physicsClientId
        p.resetSimulation(physicsClientId=pid)
        p.setPhysicsEngineParameter(enableFileCaching=0, physicsClientId=pid)

        # Set up the environment
        p.setGravity(0, 0, -10, physicsClientId=pid)
        plane_shape = p.createCollisionShape(p.GEOM_PLANE, physicsClientId=pid)
        p.createMultiBody(plane_shape, plane_shape, physicsClientId=pid)

        # Save and load creature URDF
        xml_file = f"temp_{self.sim_id}.urdf"
        with open(xml_file, "w") as f:
            f.write(cr.to_xml())
        creature_id = p.loadURDF(xml_file, [0, 0, 5.0], [0, 0, 0, 1], physicsClientId=pid)

        # Simulation loop
        for frame in range(total_frames):
            p.stepSimulation(physicsClientId=pid)

            # Update motors every 24 frames
            if frame % 24 == 0:
                motors = cr.get_motors()
                for jid in range(p.getNumJoints(creature_id, physicsClientId=pid)):
                    if jid < len(motors):
                        motor = motors[jid]
                        p.setJointMotorControl2(
                            bodyUniqueId=creature_id,
                            jointIndex=jid,
                            controlMode=p.VELOCITY_CONTROL,
                            targetVelocity=motor.get_output(),
                            force=10,
                            physicsClientId=pid,
                        )

            # Update position
            pos, _ = p.getBasePositionAndOrientation(creature_id, physicsClientId=pid)
            cr.update_position(pos)

        # Calculate fitness as maximum height
        max_height = cr.last_position[2] if cr.last_position else 0
        p.removeBody(creature_id, physicsClientId=pid)
        return max_height



class ThreadedSim():
    def __init__(self, pool_size):
        self.sims = [Simulation(i) for i in range(pool_size)]

    @staticmethod
    def static_run_creature(sim, cr, iterations):
        sim.run_creature(cr, iterations)
        return cr
    
    def eval_population(self, pop, iterations):
        """
        pop is a Population object
        iterations is frames in pybullet to run for at 240fps
        """
        pool_args = [] 
        start_ind = 0
        pool_size = len(self.sims)
        while start_ind < len(pop.creatures):
            this_pool_args = []
            for i in range(start_ind, start_ind + pool_size):
                if i == len(pop.creatures):# the end
                    break
                # work out the sim ind
                sim_ind = i % len(self.sims)
                this_pool_args.append([
                            self.sims[sim_ind], 
                            pop.creatures[i], 
                            iterations]   
                )
            pool_args.append(this_pool_args)
            start_ind = start_ind + pool_size

        new_creatures = []
        for pool_argset in pool_args:
            with Pool(pool_size) as p:
                # it works on a copy of the creatures, so receive them
                creatures = p.starmap(ThreadedSim.static_run_creature, pool_argset)
                # and now put those creatures back into the main 
                # self.creatures array
                new_creatures.extend(creatures)
        pop.creatures = new_creatures
