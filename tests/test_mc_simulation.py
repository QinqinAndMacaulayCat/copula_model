import unittest
import numpy as np
from copula_model.mc_simulation import (simulate_gaussian_copula, 
                                        simulate_t_copula, 
                                        simulate_gumbel_copula, 
                                        simulate_clayton_copula, 
                                        simulate_frank_copula)

class TestMCSimulation(unittest.TestCase):

    def setUp(self):
        self.n_paths = 1000
        self.delta_t = 1
        self.corr_matrix = np.array([[1.0, 0.5], [0.5, 1.0]])
        self.n_assets = 2
        self.n_steps = 1000
        self.random_state = 66

    
    def test_simulate_gaussian_copula(self):
        paths = simulate_gaussian_copula(self.n_paths,
                                         self.corr_matrix,
                                         self.n_assets,
                                         self.n_steps,
                                         self.random_state)
        # Check the shape of the output 
        self.assertEqual(paths.shape, (self.n_paths, self.n_steps, self.n_assets))

        # Check that the correlation is approximately correct
        sample_corr = np.corrcoef(paths[:, :, 0].flatten(), paths[:, :, 1].flatten())[0, 1]
        print(f"Sample correlation: {sample_corr}")
        print(f"Target correlation: {self.corr_matrix[0, 1]}")
        self.assertAlmostEqual(sample_corr, self.corr_matrix[0, 1], places=1)

    def test_simulate_t_copula(self):
        df = 5
        paths = simulate_t_copula(self.n_paths,
                                  self.corr_matrix,
                                  self.n_assets,
                                  self.n_steps,
                                  df,
                                  self.random_state)
        # Check the shape of the output 
        self.assertEqual(paths.shape, (self.n_paths, self.n_steps, self.n_assets))

         # Check that the correlation is approximately correct
        sample_corr = np.corrcoef(paths[:, :, 0].flatten(), paths[:, :, 1].flatten())[0, 1]
        print(f"Sample correlation (t-copula): {sample_corr}")
        print(f"Target correlation: {self.corr_matrix[0, 1]}")
        self.assertAlmostEqual(sample_corr, self.corr_matrix[0, 1], places=1)
    
    def test_simulate_clayton_copula(self):
        theta = 2.0
        paths = simulate_clayton_copula(self.n_paths,
                                        theta,
                                        self.n_assets,
                                        self.n_steps,
                                        self.random_state)
        # Check the shape of the output 
        self.assertEqual(paths.shape, (self.n_paths, self.n_steps, self.n_assets))

        # Check that values are in [0, 1]
        self.assertTrue(np.all(paths >= 0) and np.all(paths <= 1))

        # Check dependence structure by lambda
        u = paths[:, :, 0].flatten()
        v = paths[:, :, 1].flatten()
        threshold = 0.1
        lambda_empirical = np.mean((u < threshold) & (v < threshold)) / threshold
        lambda_theoretical = 2 ** (-1/theta)
        print(f"Empirical lower tail dependence (Clayton): {lambda_empirical}, Theoretical: {lambda_theoretical}")
        self.assertAlmostEqual(lambda_empirical, lambda_theoretical, places=1)

    def tearDown(self):
        return super().tearDown()
    

if __name__ == '__main__':
    unittest.main()  
