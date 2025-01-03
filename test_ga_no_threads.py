import unittest
import population
import simulation 
import genome 
import creature 
import numpy as np

class TestGA(unittest.TestCase):
    def testBasicGA(self):
        pop = population.Population(pop_size=10, 
                                    gene_count=3)
        sim = simulation.Simulation()

        for iteration in range(50):
            # Run simulation for each creature
            for cr in pop.creatures:
                sim.run_creature(cr, 2400)
            
            # Calculate fitness based on height
            fits = [cr.get_height() for cr in pop.creatures]
            links = [len(cr.get_expanded_links()) for cr in pop.creatures]
            print(iteration, "fittest:", np.round(np.max(fits), 3), 
                  "mean:", np.round(np.mean(fits), 3), 
                  "mean links", np.round(np.mean(links)), 
                  "max links", np.round(np.max(links)))
            
            # Perform selection and evolution
            fit_map = population.Population.get_fitness_map(fits)
            new_creatures = []
            for i in range(len(pop.creatures)):
                p1_ind = population.Population.select_parent(fit_map)
                p2_ind = population.Population.select_parent(fit_map)
                p1 = pop.creatures[p1_ind]
                p2 = pop.creatures[p2_ind]
                
                # Crossover and mutation
                dna = genome.Genome.crossover(p1.dna, p2.dna)
                dna = genome.Genome.point_mutate(dna, rate=0.1, amount=0.25)
                dna = genome.Genome.shrink_mutate(dna, rate=0.25)
                dna = genome.Genome.grow_mutate(dna, rate=0.1)
                
                # Create a new creature
                cr = creature.Creature(1)
                cr.update_dna(dna)
                new_creatures.append(cr)
            
            # Elitism: Keep the best creature
            max_fit = np.max(fits)
            for cr in pop.creatures:
                if cr.get_height() == max_fit:
                    new_cr = creature.Creature(1)
                    new_cr.update_dna(cr.dna)
                    new_creatures[0] = new_cr
                    filename = "tests/elite_" + str(iteration) + ".csv"
                    genome.Genome.to_csv(cr.dna, filename)
                    break
            
            # Update population with new generation
            pop.creatures = new_creatures

        # Ensure at least one creature shows fitness > 0
        self.assertNotEqual(fits[0], 0)

unittest.main()
