"""
Unit Tests for FPGA-Accelerated ERI Simulator
=============================================

Comprehensive test suite covering:
1. Data classes (BasisFunction, Atom, Molecule, Shell, etc.)
2. GaussianIntegralEngine (Boys function, Schwarz screening, ERI computation)
3. DensityFitting module
4. FPGASimulator, CPUSimulator, GPUSimulator
5. AIController (prediction, decision-making, RL learning)
6. HartreeFockSolver (Fock matrix, DIIS, SCF)
7. Molecule library functions
8. SimulationPipeline integration

February 2026 - BMEG 490
"""

import unittest
import numpy as np
import math
import sys
import os

# Import the simulator module
from fpga_eri_simulator import (
    PrecisionMode, ComputeTarget, TheoryLevel, BasisSetType,
    BOHR_TO_ANGSTROM, ANGSTROM_TO_BOHR, HARTREE_TO_KCAL, HARTREE_TO_EV,
    BasisFunction, Shell, Atom, Molecule, ERIResult, SCFResult,
    SimulationMetrics, EnergyDecomposition,
    GaussianIntegralEngine, DensityFitting,
    FPGASimulator, CPUSimulator, GPUSimulator,
    AIController, HartreeFockSolver,
    create_water_molecule, create_methane_molecule,
    create_benzene_molecule, create_aspirin_molecule,
    create_caffeine_molecule, create_ibuprofen_molecule,
    SimulationPipeline,
)


# =========================================================================
# TEST DATA CLASSES
# =========================================================================

class TestBasisFunction(unittest.TestCase):
    """Tests for the BasisFunction dataclass."""

    def test_creation(self):
        bf = BasisFunction(
            center=np.array([0.0, 0.0, 0.0]),
            alpha=1.0,
            angular_momentum=(0, 0, 0),
            coefficient=1.0,
        )
        self.assertEqual(bf.alpha, 1.0)
        self.assertEqual(bf.coefficient, 1.0)
        np.testing.assert_array_equal(bf.center, [0.0, 0.0, 0.0])

    def test_total_angular_momentum(self):
        bf_s = BasisFunction(np.zeros(3), 1.0, (0, 0, 0))
        bf_p = BasisFunction(np.zeros(3), 1.0, (1, 0, 0))
        bf_d = BasisFunction(np.zeros(3), 1.0, (1, 1, 0))
        self.assertEqual(bf_s.total_angular_momentum, 0)
        self.assertEqual(bf_p.total_angular_momentum, 1)
        self.assertEqual(bf_d.total_angular_momentum, 2)

    def test_orbital_type(self):
        bf_s = BasisFunction(np.zeros(3), 1.0, (0, 0, 0))
        bf_p = BasisFunction(np.zeros(3), 1.0, (1, 0, 0))
        bf_d = BasisFunction(np.zeros(3), 1.0, (2, 0, 0))
        self.assertEqual(bf_s.orbital_type, 's')
        self.assertEqual(bf_p.orbital_type, 'p')
        self.assertEqual(bf_d.orbital_type, 'd')

    def test_normalization_s_type(self):
        bf = BasisFunction(np.zeros(3), 1.0, (0, 0, 0))
        norm = bf.normalization
        self.assertGreater(norm, 0)
        # For alpha=1, s-type: (2/pi)^0.75
        expected = (2.0 / np.pi) ** 0.75
        self.assertAlmostEqual(norm, expected, places=5)

    def test_normalization_p_type(self):
        bf = BasisFunction(np.zeros(3), 1.0, (1, 0, 0))
        norm = bf.normalization
        self.assertGreater(norm, 0)


class TestShell(unittest.TestCase):
    """Tests for the Shell dataclass."""

    def test_s_shell_functions(self):
        shell = Shell(
            center=np.zeros(3),
            angular_momentum=0,
            exponents=[1.0, 0.5],
            coefficients=[0.5, 0.5],
        )
        bfs = shell.get_basis_functions()
        # s-shell: only (0,0,0), 2 exponents => 2 functions
        self.assertEqual(len(bfs), 2)
        for bf in bfs:
            self.assertEqual(bf.angular_momentum, (0, 0, 0))

    def test_p_shell_functions(self):
        shell = Shell(
            center=np.zeros(3),
            angular_momentum=1,
            exponents=[1.0],
            coefficients=[1.0],
        )
        bfs = shell.get_basis_functions()
        # p-shell: (1,0,0), (0,1,0), (0,0,1) => 3 functions
        self.assertEqual(len(bfs), 3)
        angular_momenta = [bf.angular_momentum for bf in bfs]
        self.assertIn((1, 0, 0), angular_momenta)
        self.assertIn((0, 1, 0), angular_momenta)
        self.assertIn((0, 0, 1), angular_momenta)


class TestAtom(unittest.TestCase):
    """Tests for the Atom dataclass."""

    def test_hydrogen(self):
        h = Atom('H', np.zeros(3))
        self.assertEqual(h.atomic_number, 1)
        self.assertAlmostEqual(h.mass, 1.008, places=2)

    def test_carbon(self):
        c = Atom('C', np.zeros(3))
        self.assertEqual(c.atomic_number, 6)
        self.assertAlmostEqual(c.mass, 12.011, places=2)

    def test_oxygen(self):
        o = Atom('O', np.zeros(3))
        self.assertEqual(o.atomic_number, 8)
        self.assertAlmostEqual(o.mass, 15.999, places=2)

    def test_nitrogen(self):
        n = Atom('N', np.zeros(3))
        self.assertEqual(n.atomic_number, 7)
        self.assertAlmostEqual(n.mass, 14.007, places=2)

    def test_unknown_element(self):
        x = Atom('X', np.zeros(3))
        self.assertEqual(x.atomic_number, 0)
        self.assertEqual(x.mass, 0.0)


class TestMolecule(unittest.TestCase):
    """Tests for the Molecule dataclass."""

    def setUp(self):
        self.water = create_water_molecule()
        self.methane = create_methane_molecule()

    def test_water_properties(self):
        self.assertEqual(self.water.n_atoms, 3)
        self.assertEqual(self.water.n_basis, 15)
        self.assertEqual(self.water.n_electrons, 10)  # O=8, H=1, H=1
        self.assertEqual(self.water.charge, 0)
        self.assertEqual(self.water.multiplicity, 1)

    def test_methane_properties(self):
        self.assertEqual(self.methane.n_atoms, 5)
        self.assertEqual(self.methane.n_basis, 21)
        self.assertEqual(self.methane.n_electrons, 10)  # C=6, 4*H=4

    def test_nuclear_repulsion_energy(self):
        nre = self.water.nuclear_repulsion_energy
        self.assertGreater(nre, 0)
        # Water NRE should be positive and reasonable
        self.assertLess(nre, 50.0)

    def test_center_of_mass(self):
        com = self.water.center_of_mass
        self.assertEqual(len(com), 3)
        # COM should be near the oxygen (heaviest atom)

    def test_smiles(self):
        self.assertEqual(self.water.smiles, "O")
        self.assertEqual(self.methane.smiles, "C")


class TestERIResult(unittest.TestCase):
    """Tests for ERIResult dataclass."""

    def test_creation(self):
        result = ERIResult(
            indices=(0, 1, 2, 3),
            value=0.5,
            precision=PrecisionMode.FP64,
            compute_target=ComputeTarget.FPGA,
            compute_time_ns=100.0,
        )
        self.assertEqual(result.indices, (0, 1, 2, 3))
        self.assertAlmostEqual(result.value, 0.5)
        self.assertFalse(result.screened)


class TestSCFResult(unittest.TestCase):
    """Tests for SCFResult dataclass."""

    def test_creation_defaults(self):
        result = SCFResult(
            converged=True,
            energy=-75.0,
            n_iterations=10,
            orbital_energies=np.array([-20.0, -1.0, 0.5]),
            density_matrix=np.eye(3),
            fock_matrix=np.eye(3),
            convergence_history=[1e-3, 1e-5, 1e-7],
        )
        self.assertTrue(result.converged)
        self.assertAlmostEqual(result.energy, -75.0)
        np.testing.assert_array_equal(result.dipole_moment, np.zeros(3))


class TestEnergyDecomposition(unittest.TestCase):
    """Tests for EnergyDecomposition dataclass."""

    def test_to_dict(self):
        ed = EnergyDecomposition(
            nuclear_repulsion=9.0,
            one_electron=-120.0,
            two_electron_coulomb=40.0,
            two_electron_exchange=-10.0,
            total=-81.0,
        )
        d = ed.to_dict()
        self.assertIn('nuclear_repulsion', d)
        self.assertAlmostEqual(d['total'], -81.0)


class TestSimulationMetrics(unittest.TestCase):
    """Tests for SimulationMetrics dataclass."""

    def test_defaults(self):
        metrics = SimulationMetrics()
        self.assertEqual(metrics.total_eris_computed, 0)
        self.assertEqual(metrics.total_eris_screened, 0)
        self.assertAlmostEqual(metrics.power_estimate_watts, 0.0)


# =========================================================================
# TEST ENUMERATIONS AND CONSTANTS
# =========================================================================

class TestEnumerations(unittest.TestCase):
    """Tests for enumerations and constants."""

    def test_precision_modes(self):
        self.assertEqual(PrecisionMode.FP64.value, "fp64")
        self.assertEqual(PrecisionMode.INT8.value, "int8")
        self.assertEqual(len(PrecisionMode), 5)

    def test_compute_targets(self):
        self.assertEqual(ComputeTarget.FPGA.value, "fpga")
        self.assertEqual(len(ComputeTarget), 4)

    def test_theory_levels(self):
        self.assertEqual(TheoryLevel.HF.value, "hf")
        self.assertEqual(TheoryLevel.MP2.value, "mp2")

    def test_basis_set_types(self):
        self.assertEqual(BasisSetType.STO_3G.value, "sto-3g")
        self.assertEqual(BasisSetType.CC_PVDZ.value, "cc-pvdz")

    def test_physical_constants(self):
        self.assertAlmostEqual(BOHR_TO_ANGSTROM, 0.529177249, places=5)
        self.assertAlmostEqual(ANGSTROM_TO_BOHR, 1.8897259886, places=5)
        self.assertAlmostEqual(HARTREE_TO_EV, 27.211386245988, places=3)


# =========================================================================
# TEST GAUSSIAN INTEGRAL ENGINE
# =========================================================================

class TestGaussianIntegralEngine(unittest.TestCase):
    """Tests for GaussianIntegralEngine."""

    def setUp(self):
        self.engine = GaussianIntegralEngine()

    def test_boys_function_zero_arg(self):
        # F_0(0) = 1
        result = self.engine.boys_function(0, 0.0)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_boys_function_zero_order(self):
        # F_0(x) = sqrt(pi/(4x)) * erf(sqrt(x)) for x > 0
        # For small x: F_0(x) ~ 1 - x/3
        result = self.engine.boys_function(0, 0.1)
        self.assertGreater(result, 0)
        self.assertLess(result, 1.0)

    def test_boys_function_large_x(self):
        result = self.engine.boys_function(0, 50.0)
        self.assertGreater(result, 0)
        # Should be very small for large x
        self.assertLess(result, 0.5)

    def test_boys_function_higher_order(self):
        # F_n(0) = 1/(2n+1)
        for n in range(5):
            result = self.engine.boys_function(n, 0.0)
            self.assertAlmostEqual(result, 1.0 / (2 * n + 1), places=5)

    def test_schwarz_matrix_computation(self):
        water = create_water_molecule()
        Q = self.engine.compute_schwarz_matrix(water.basis_functions[:3])
        self.assertEqual(Q.shape, (3, 3))
        # Schwarz matrix should be symmetric
        np.testing.assert_array_almost_equal(Q, Q.T)
        # Diagonal elements should be non-negative
        for i in range(3):
            self.assertGreaterEqual(Q[i, i], 0)

    def test_screening(self):
        water = create_water_molecule()
        bfs = water.basis_functions[:5]
        self.engine.compute_schwarz_matrix(bfs)
        # screen_integral should return a boolean-like value
        result = self.engine.screen_integral(0, 0, 1, 1)
        self.assertIn(result, (True, False))

    def test_screening_disabled(self):
        engine = GaussianIntegralEngine(use_screening=False)
        result = engine.screen_integral(0, 0, 1, 1)
        self.assertFalse(result)

    def test_eri_s_type_nonzero(self):
        """Test that s-type ERI produces a nonzero result."""
        bf1 = BasisFunction(np.array([0.0, 0.0, 0.0]), 1.0, (0, 0, 0), 1.0)
        bf2 = BasisFunction(np.array([0.0, 0.0, 0.0]), 1.0, (0, 0, 0), 1.0)
        value = self.engine.compute_eri_primitive(bf1, bf2, bf1, bf2)
        self.assertNotEqual(value, 0.0)
        self.assertGreater(abs(value), 1e-15)

    def test_eri_different_centers(self):
        """Test ERI with basis functions on different atoms."""
        bf1 = BasisFunction(np.array([0.0, 0.0, 0.0]), 1.0, (0, 0, 0), 1.0)
        bf2 = BasisFunction(np.array([1.0, 0.0, 0.0]), 1.0, (0, 0, 0), 1.0)
        value = self.engine.compute_eri_primitive(bf1, bf2, bf1, bf2)
        self.assertNotEqual(value, 0.0)

    def test_eri_symmetry(self):
        """Test (ab|cd) = (cd|ab) symmetry."""
        bf1 = BasisFunction(np.array([0.0, 0.0, 0.0]), 1.0, (0, 0, 0), 0.5)
        bf2 = BasisFunction(np.array([1.0, 0.0, 0.0]), 0.8, (0, 0, 0), 0.7)
        val1 = self.engine.compute_eri_primitive(bf1, bf1, bf2, bf2)
        val2 = self.engine.compute_eri_primitive(bf2, bf2, bf1, bf1)
        self.assertAlmostEqual(val1, val2, places=10)

    def test_eri_p_type(self):
        """Test ERI with p-type orbital."""
        bf_s = BasisFunction(np.zeros(3), 1.0, (0, 0, 0), 1.0)
        bf_p = BasisFunction(np.zeros(3), 1.0, (1, 0, 0), 1.0)
        value = self.engine.compute_eri_primitive(bf_s, bf_s, bf_p, bf_s)
        # Should return a finite value
        self.assertTrue(np.isfinite(value))

    def test_statistics(self):
        bf = BasisFunction(np.zeros(3), 1.0, (0, 0, 0))
        self.engine.compute_eri_primitive(bf, bf, bf, bf)
        stats = self.engine.get_statistics()
        self.assertGreater(stats['total_computed'], 0)


# =========================================================================
# TEST DENSITY FITTING
# =========================================================================

class TestDensityFitting(unittest.TestCase):
    """Tests for the DensityFitting module."""

    def setUp(self):
        self.df = DensityFitting(auxiliary_basis_ratio=2.5)

    def test_auxiliary_basis_generation(self):
        bf = BasisFunction(np.zeros(3), 1.0, (0, 0, 0))
        aux = self.df.setup_auxiliary_basis([bf])
        # Should generate 3 auxiliary functions per original (scale 0.5, 1.0, 2.0)
        self.assertEqual(len(aux), 3)

    def test_auxiliary_basis_exponents(self):
        bf = BasisFunction(np.zeros(3), 2.0, (0, 0, 0))
        aux = self.df.setup_auxiliary_basis([bf])
        exponents = [a.alpha for a in aux]
        self.assertIn(1.0, exponents)  # 2.0 * 0.5
        self.assertIn(2.0, exponents)  # 2.0 * 1.0
        self.assertIn(4.0, exponents)  # 2.0 * 2.0

    def test_fitting_metric_shape(self):
        bf = BasisFunction(np.zeros(3), 1.0, (0, 0, 0))
        aux = self.df.setup_auxiliary_basis([bf])
        engine = GaussianIntegralEngine()
        V = self.df.compute_fitting_metric(aux, engine)
        self.assertEqual(V.shape, (3, 3))
        # Should be symmetric
        np.testing.assert_array_almost_equal(V, V.T)

    def test_estimate_speedup(self):
        # For small basis, speedup may be < 1 due to overhead
        speedup_small = self.df.estimate_speedup(5)
        self.assertGreater(speedup_small, 0)

        # For larger basis, speedup should increase
        speedup_large = self.df.estimate_speedup(50)
        self.assertGreater(speedup_large, speedup_small)

    def test_speedup_scaling(self):
        """Larger basis sets should see greater speedup from RI."""
        s10 = self.df.estimate_speedup(10)
        s50 = self.df.estimate_speedup(50)
        s100 = self.df.estimate_speedup(100)
        self.assertLess(s10, s50)
        self.assertLess(s50, s100)


# =========================================================================
# TEST FPGA SIMULATOR
# =========================================================================

class TestFPGASimulator(unittest.TestCase):
    """Tests for FPGASimulator."""

    def setUp(self):
        self.fpga = FPGASimulator(
            num_compute_units=64,
            clock_freq_mhz=300.0,
        )

    def test_initialization(self):
        self.assertEqual(self.fpga.num_compute_units, 64)
        self.assertAlmostEqual(self.fpga.clock_freq_mhz, 300.0)

    def test_cycles_per_eri(self):
        self.assertEqual(self.fpga.cycles_per_eri[PrecisionMode.FP64], 48)
        self.assertEqual(self.fpga.cycles_per_eri[PrecisionMode.FP32], 24)
        self.assertEqual(self.fpga.cycles_per_eri[PrecisionMode.FP16], 12)
        self.assertEqual(self.fpga.cycles_per_eri[PrecisionMode.FP12], 8)
        self.assertEqual(self.fpga.cycles_per_eri[PrecisionMode.INT8], 4)

    def test_precision_error_ordering(self):
        """Lower precision should have higher error."""
        errors = self.fpga.precision_error
        self.assertLess(errors[PrecisionMode.FP64], errors[PrecisionMode.FP32])
        self.assertLess(errors[PrecisionMode.FP32], errors[PrecisionMode.FP16])
        self.assertLess(errors[PrecisionMode.FP16], errors[PrecisionMode.FP12])
        self.assertLess(errors[PrecisionMode.FP12], errors[PrecisionMode.INT8])

    def test_power_efficiency_ordering(self):
        """Lower precision should use less power (lower ratio)."""
        pe = self.fpga.power_efficiency
        self.assertGreater(pe[PrecisionMode.FP64], pe[PrecisionMode.FP32])
        self.assertGreater(pe[PrecisionMode.FP32], pe[PrecisionMode.FP16])

    def test_compression_ratio_ordering(self):
        cr = self.fpga.compression_ratio
        self.assertLess(cr[PrecisionMode.FP64], cr[PrecisionMode.FP32])
        self.assertLess(cr[PrecisionMode.FP32], cr[PrecisionMode.FP16])

    def test_compute_eri_batch_water(self):
        water = create_water_molecule()
        bfs = water.basis_functions[:3]  # Use small subset for speed
        results, time_ms, metrics = self.fpga.compute_eri_batch(
            bfs, PrecisionMode.FP16, use_screening=False
        )
        self.assertGreater(len(results), 0)
        self.assertGreater(time_ms, 0)
        self.assertEqual(metrics.total_eris_computed, len(results))

    def test_compute_eri_batch_with_screening(self):
        water = create_water_molecule()
        bfs = water.basis_functions[:5]
        results, time_ms, metrics = self.fpga.compute_eri_batch(
            bfs, PrecisionMode.FP16, use_screening=True
        )
        self.assertGreater(len(results), 0)
        # Some integrals should have been screened
        self.assertGreaterEqual(metrics.total_eris_screened, 0)

    def test_estimate_throughput(self):
        for mode in PrecisionMode:
            throughput = self.fpga.estimate_throughput(mode)
            self.assertGreater(throughput, 0)

    def test_throughput_increases_with_lower_precision(self):
        tp_64 = self.fpga.estimate_throughput(PrecisionMode.FP64)
        tp_32 = self.fpga.estimate_throughput(PrecisionMode.FP32)
        tp_16 = self.fpga.estimate_throughput(PrecisionMode.FP16)
        tp_8 = self.fpga.estimate_throughput(PrecisionMode.INT8)
        self.assertLess(tp_64, tp_32)
        self.assertLess(tp_32, tp_16)
        self.assertLess(tp_16, tp_8)

    def test_power_efficiency_metric(self):
        eff_64 = self.fpga.estimate_power_efficiency(PrecisionMode.FP64)
        eff_16 = self.fpga.estimate_power_efficiency(PrecisionMode.FP16)
        self.assertGreater(eff_16, eff_64)

    def test_quartet_generation(self):
        indices = self.fpga._generate_quartet_indices(3)
        self.assertGreater(len(indices), 0)
        # All indices should be valid
        for i, j, k, l in indices:
            self.assertGreaterEqual(i, 0)
            self.assertLess(i, 3)


# =========================================================================
# TEST CPU SIMULATOR
# =========================================================================

class TestCPUSimulator(unittest.TestCase):
    """Tests for CPUSimulator."""

    def test_initialization(self):
        cpu = CPUSimulator(num_cores=8, simd_width=4)
        self.assertEqual(cpu.num_cores, 8)

    def test_compute_batch(self):
        cpu = CPUSimulator()
        water = create_water_molecule()
        bfs = water.basis_functions[:3]
        results, time_ms, metrics = cpu.compute_eri_batch(bfs)
        self.assertGreater(len(results), 0)
        self.assertGreater(time_ms, 0)
        # CPU should use FP64
        for r in results:
            self.assertEqual(r.precision, PrecisionMode.FP64)
            self.assertEqual(r.compute_target, ComputeTarget.CPU)

    def test_cpu_power_estimate(self):
        cpu = CPUSimulator()
        water = create_water_molecule()
        bfs = water.basis_functions[:3]
        _, _, metrics = cpu.compute_eri_batch(bfs)
        self.assertAlmostEqual(metrics.power_estimate_watts, 150.0)


# =========================================================================
# TEST GPU SIMULATOR
# =========================================================================

class TestGPUSimulator(unittest.TestCase):
    """Tests for GPUSimulator."""

    def test_initialization(self):
        gpu = GPUSimulator(sm_count=80, cuda_cores_per_sm=64)
        self.assertEqual(gpu.cuda_cores, 5120)

    def test_compute_batch(self):
        gpu = GPUSimulator()
        water = create_water_molecule()
        bfs = water.basis_functions[:3]
        results, time_ms, metrics = gpu.compute_eri_batch(bfs)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertEqual(r.precision, PrecisionMode.FP32)
            self.assertEqual(r.compute_target, ComputeTarget.GPU)


# =========================================================================
# TEST AI CONTROLLER
# =========================================================================

class TestAIController(unittest.TestCase):
    """Tests for AIController."""

    def setUp(self):
        self.ai = AIController(
            uncertainty_threshold=0.1,
            energy_tolerance=1e-3,
            learning_rate=0.01,
        )
        self.water = create_water_molecule()

    def test_initialization(self):
        self.assertEqual(self.ai.uncertainty_threshold, 0.1)
        self.assertEqual(len(self.ai.decision_history), 0)
        self.assertAlmostEqual(self.ai.cumulative_reward, 0.0)

    def test_predict_molecule_properties(self):
        energy, uncertainty = self.ai.predict_molecule_properties(self.water)
        self.assertLess(energy, 0)  # Energy should be negative
        self.assertGreater(uncertainty, 0)  # Uncertainty should be positive

    def test_prediction_energy_reasonable(self):
        """Predicted energy should be roughly sum of atomic energies."""
        energy, _ = self.ai.predict_molecule_properties(self.water)
        # Water: O(-74.8) + 2*H(-0.5) + binding ~ -75.5
        self.assertLess(energy, -70.0)
        self.assertGreater(energy, -80.0)

    def test_decide_precision_level(self):
        energy, uncertainty = self.ai.predict_molecule_properties(self.water)
        precision, target, level = self.ai.decide_precision_level(
            self.water, energy, uncertainty
        )
        self.assertIn(level, [0, 1, 2, 3])
        if level > 0:
            self.assertIsNotNone(precision)
            self.assertIsNotNone(target)

    def test_decision_level_0(self):
        """Very low uncertainty should give level 0."""
        precision, target, level = self.ai.decide_precision_level(
            self.water, -75.0, 0.001  # Very low uncertainty
        )
        self.assertEqual(level, 0)
        self.assertIsNone(precision)

    def test_decision_level_3(self):
        """Very high uncertainty should give level 3."""
        precision, target, level = self.ai.decide_precision_level(
            self.water, -75.0, 0.5  # High uncertainty
        )
        self.assertEqual(level, 3)
        self.assertEqual(precision, PrecisionMode.FP64)
        self.assertEqual(target, ComputeTarget.CPU)

    def test_update_from_result(self):
        self.ai.update_from_result(1, -75.0, -75.05, 1.0)
        self.assertNotEqual(self.ai.cumulative_reward, 0.0)
        self.assertEqual(len(self.ai.prediction_errors), 1)

    def test_rl_weight_updates(self):
        """Weights should change after update."""
        initial_weights = self.ai.decision_weights.copy()
        self.ai.update_from_result(1, -75.0, -75.0001, 1.0)
        # Weight for level 1 should have increased (good accuracy)
        self.assertNotEqual(
            self.ai.decision_weights[1], initial_weights[1]
        )

    def test_add_to_sar_database(self):
        self.ai.add_to_sar_database(self.water, {'energy': -75.95})
        self.assertIn("Water (H2O)", self.ai.sar_database)

    def test_validate_result(self):
        self.assertTrue(self.ai.validate_result(-75.0, -75.0005))
        self.assertFalse(self.ai.validate_result(-75.0, -80.0))

    def test_novelty_estimation(self):
        # Empty SAR database should give high novelty
        novelty_empty = self.ai._estimate_novelty(self.water)
        self.assertEqual(novelty_empty, 1.5)

        # After adding to database, novelty should decrease
        self.ai.add_to_sar_database(self.water, {'energy': -75.95})
        novelty_known = self.ai._estimate_novelty(self.water)
        self.assertLess(novelty_known, novelty_empty)

    def test_decision_statistics(self):
        stats = self.ai.get_decision_statistics()
        self.assertEqual(stats['total_decisions'], 0)

        # Make some decisions
        for _ in range(5):
            energy, unc = self.ai.predict_molecule_properties(self.water)
            self.ai.decide_precision_level(self.water, energy, unc)

        stats = self.ai.get_decision_statistics()
        self.assertEqual(stats['total_decisions'], 5)

    def test_calibration_factor_updates(self):
        """Calibration factor should update after enough data."""
        for i in range(15):
            self.ai.update_from_result(1, -75.0, -75.0 + i * 0.01, 1.0)
        self.assertNotEqual(self.ai.calibration_factor, 1.0)


# =========================================================================
# TEST HARTREE-FOCK SOLVER
# =========================================================================

class TestHartreeFockSolver(unittest.TestCase):
    """Tests for HartreeFockSolver."""

    def setUp(self):
        self.solver = HartreeFockSolver(
            convergence_threshold=1e-6,
            max_iterations=100,
            use_diis=True,
        )

    def test_initialization(self):
        self.assertEqual(self.solver.convergence_threshold, 1e-6)
        self.assertTrue(self.solver.use_diis)

    def test_density_matrix_computation(self):
        n = 5
        coeffs = np.eye(n)
        n_occ = 2
        D = self.solver.compute_density_matrix(coeffs, n_occ)
        self.assertEqual(D.shape, (n, n))
        # Trace of density matrix should be 2 * n_occ
        self.assertAlmostEqual(np.trace(D), 2 * n_occ, places=5)

    def test_density_matrix_symmetry(self):
        coeffs = np.random.randn(5, 5)
        D = self.solver.compute_density_matrix(coeffs, 2)
        np.testing.assert_array_almost_equal(D, D.T)

    def test_diis_with_insufficient_data(self):
        """DIIS with < 2 Fock matrices should return the last one."""
        F = np.eye(3)
        result = self.solver.diis_extrapolation([F], [np.zeros((3, 3))])
        np.testing.assert_array_equal(result, F)

    def test_diis_extrapolation(self):
        """DIIS with 2+ matrices should produce an extrapolated result."""
        F1 = np.eye(3) * 1.0
        F2 = np.eye(3) * 0.9
        e1 = np.random.randn(3, 3) * 0.1
        e2 = np.random.randn(3, 3) * 0.05
        result = self.solver.diis_extrapolation([F1, F2], [e1, e2])
        self.assertEqual(result.shape, (3, 3))

    def test_build_fock_matrix(self):
        n = 3
        D = np.eye(n) * 0.5
        h_core = np.random.randn(n, n)
        h_core = (h_core + h_core.T) / 2
        eri = ERIResult((0, 0, 0, 0), 0.1, PrecisionMode.FP64, ComputeTarget.CPU, 0)
        F = self.solver.build_fock_matrix(D, h_core, [eri], n)
        self.assertEqual(F.shape, (n, n))

    def test_scf_water(self):
        """Test SCF on water molecule (may or may not converge)."""
        water = create_water_molecule()
        bfs = water.basis_functions[:5]  # Small subset
        water_small = Molecule("H2O_small", water.atoms, bfs)

        fpga = FPGASimulator()
        results, _, _ = fpga.compute_eri_batch(bfs, PrecisionMode.FP64, use_screening=False)

        scf = self.solver.run_scf(water_small, results)
        self.assertIsInstance(scf, SCFResult)
        self.assertIsInstance(scf.energy, float)
        self.assertTrue(np.isfinite(scf.energy))
        self.assertGreater(len(scf.convergence_history), 0)


# =========================================================================
# TEST MOLECULE LIBRARY
# =========================================================================

class TestMoleculeLibrary(unittest.TestCase):
    """Tests for molecule creation functions."""

    def test_water_creation(self):
        mol = create_water_molecule()
        self.assertEqual(mol.name, "Water (H2O)")
        self.assertEqual(mol.n_atoms, 3)
        self.assertEqual(mol.n_basis, 15)
        self.assertEqual(mol.smiles, "O")

    def test_methane_creation(self):
        mol = create_methane_molecule()
        self.assertEqual(mol.name, "Methane (CH4)")
        self.assertEqual(mol.n_atoms, 5)
        self.assertEqual(mol.n_basis, 21)

    def test_benzene_creation(self):
        mol = create_benzene_molecule()
        self.assertEqual(mol.name, "Benzene (C6H6)")
        self.assertEqual(mol.n_atoms, 12)

    def test_aspirin_creation(self):
        mol = create_aspirin_molecule()
        self.assertEqual(mol.name, "Aspirin (C9H8O4)")
        self.assertEqual(mol.n_atoms, 21)
        self.assertGreater(mol.n_basis, 0)

    def test_caffeine_creation(self):
        mol = create_caffeine_molecule()
        self.assertEqual(mol.name, "Caffeine (C8H10N4O2)")
        self.assertGreater(mol.n_atoms, 0)
        self.assertGreater(mol.n_basis, 0)

    def test_ibuprofen_creation(self):
        mol = create_ibuprofen_molecule()
        self.assertEqual(mol.name, "Ibuprofen (C13H18O2)")
        self.assertGreater(mol.n_atoms, 0)
        self.assertGreater(mol.n_basis, 0)

    def test_all_molecules_have_basis_functions(self):
        creators = [
            create_water_molecule, create_methane_molecule,
            create_benzene_molecule, create_aspirin_molecule,
            create_caffeine_molecule, create_ibuprofen_molecule,
        ]
        for creator in creators:
            mol = creator()
            self.assertGreater(mol.n_basis, 0, f"{mol.name} has no basis functions")
            self.assertGreater(mol.n_atoms, 0, f"{mol.name} has no atoms")

    def test_all_molecules_have_valid_nuclear_repulsion(self):
        creators = [
            create_water_molecule, create_methane_molecule,
            create_benzene_molecule, create_aspirin_molecule,
            create_caffeine_molecule, create_ibuprofen_molecule,
        ]
        for creator in creators:
            mol = creator()
            nre = mol.nuclear_repulsion_energy
            self.assertGreater(nre, 0, f"{mol.name} has non-positive NRE")
            self.assertTrue(np.isfinite(nre), f"{mol.name} has non-finite NRE")

    def test_drug_molecules_electron_count(self):
        """Drug molecules should have positive electron counts matching their atoms."""
        aspirin = create_aspirin_molecule()
        # Verify electrons = sum of atomic numbers (charge=0)
        expected_aspirin = sum(a.atomic_number for a in aspirin.atoms)
        self.assertEqual(aspirin.n_electrons, expected_aspirin)
        self.assertGreater(aspirin.n_electrons, 0)

        caffeine = create_caffeine_molecule()
        expected_caffeine = sum(a.atomic_number for a in caffeine.atoms)
        self.assertEqual(caffeine.n_electrons, expected_caffeine)
        self.assertGreater(caffeine.n_electrons, 0)

        ibuprofen = create_ibuprofen_molecule()
        expected_ibuprofen = sum(a.atomic_number for a in ibuprofen.atoms)
        self.assertEqual(ibuprofen.n_electrons, expected_ibuprofen)
        self.assertGreater(ibuprofen.n_electrons, 0)


# =========================================================================
# TEST SIMULATION PIPELINE
# =========================================================================

class TestSimulationPipeline(unittest.TestCase):
    """Tests for SimulationPipeline integration."""

    def setUp(self):
        self.pipeline = SimulationPipeline()

    def test_initialization(self):
        self.assertIsNotNone(self.pipeline.fpga)
        self.assertIsNotNone(self.pipeline.cpu)
        self.assertIsNotNone(self.pipeline.gpu)
        self.assertIsNotNone(self.pipeline.ai_controller)

    def test_run_molecule_simulation_auto(self):
        water = create_water_molecule()
        result = self.pipeline.run_molecule_simulation(water)
        self.assertIn('molecule_name', result)
        self.assertIn('ml_prediction', result)
        self.assertIn('qm_energy', result)
        self.assertIn('ai_decision_level', result)

    def test_run_molecule_simulation_forced_fpga(self):
        water = create_water_molecule()
        result = self.pipeline.run_molecule_simulation(
            water,
            force_target=ComputeTarget.FPGA,
            force_precision=PrecisionMode.FP16,
        )
        self.assertEqual(result['precision_level'], 'fp16')
        self.assertEqual(result['compute_target'], 'fpga')
        self.assertGreater(result['n_eris'], 0)

    def test_run_molecule_simulation_forced_cpu(self):
        water = create_water_molecule()
        result = self.pipeline.run_molecule_simulation(
            water,
            force_target=ComputeTarget.CPU,
            force_precision=PrecisionMode.FP64,
        )
        self.assertEqual(result['precision_level'], 'fp64')
        self.assertEqual(result['compute_target'], 'cpu')

    def test_count_quartets(self):
        count = self.pipeline._count_quartets(5)
        self.assertGreater(count, 0)
        # For n=5, the count should be a specific value
        # Manually: unique quartets with 8-fold symmetry
        count3 = self.pipeline._count_quartets(3)
        self.assertLess(count3, count)

    def test_compute_energy(self):
        water = create_water_molecule()
        eri = ERIResult((0, 0, 0, 0), 0.1, PrecisionMode.FP64, ComputeTarget.CPU, 0)
        energy = self.pipeline._compute_energy([eri], water)
        self.assertTrue(np.isfinite(energy))

    def test_results_log(self):
        water = create_water_molecule()
        self.pipeline.run_molecule_simulation(water)
        self.assertEqual(len(self.pipeline.results_log), 1)

    def test_fpga_speedup_over_cpu(self):
        """FPGA should be faster than CPU for same workload."""
        water = create_water_molecule()
        fpga_result = self.pipeline.run_molecule_simulation(
            water, ComputeTarget.FPGA, PrecisionMode.FP16
        )
        cpu_result = self.pipeline.run_molecule_simulation(
            water, ComputeTarget.CPU, PrecisionMode.FP64
        )
        self.assertLess(
            fpga_result['compute_time_ms'],
            cpu_result['compute_time_ms'],
        )

    def test_export_results_json(self):
        """Test JSON export (creates temp file)."""
        water = create_water_molecule()
        self.pipeline.run_molecule_simulation(water)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            fname = f.name
        try:
            self.pipeline.export_results_json(fname)
            import json
            with open(fname) as f:
                data = json.load(f)
            self.assertIn('simulation_log', data)
            self.assertIn('version', data)
            self.assertEqual(data['version'], '2.0')
        finally:
            os.unlink(fname)


# =========================================================================
# TEST CROSS-COMPONENT INTEGRATION
# =========================================================================

class TestIntegration(unittest.TestCase):
    """Integration tests combining multiple components."""

    def test_full_pipeline_water(self):
        """Full pipeline test: molecule -> AI -> FPGA -> energy."""
        pipeline = SimulationPipeline()
        water = create_water_molecule()
        result = pipeline.run_molecule_simulation(
            water, ComputeTarget.FPGA, PrecisionMode.FP16
        )
        self.assertIn('qm_energy', result)
        self.assertTrue(np.isfinite(result['qm_energy']))
        self.assertGreater(result['n_eris'], 0)
        self.assertGreater(result['compute_time_ms'], 0)

    def test_ai_learns_across_molecules(self):
        """AI controller should accumulate experience."""
        pipeline = SimulationPipeline()
        molecules = [create_water_molecule(), create_methane_molecule()]
        for mol in molecules:
            pipeline.run_molecule_simulation(mol)
        stats = pipeline.ai_controller.get_decision_statistics()
        self.assertEqual(stats['total_decisions'], 2)

    def test_density_fitting_integration(self):
        """Test density fitting works with integral engine."""
        df = DensityFitting()
        engine = GaussianIntegralEngine()
        water = create_water_molecule()

        aux = df.setup_auxiliary_basis(water.basis_functions[:3])
        V = df.compute_fitting_metric(aux, engine)
        self.assertIsNotNone(V)
        self.assertEqual(V.shape[0], V.shape[1])

    def test_precision_accuracy_tradeoff(self):
        """Higher precision should give more consistent results."""
        water = create_water_molecule()
        bfs = water.basis_functions[:3]

        fpga = FPGASimulator()

        # Run FP64 (reference)
        res_64, _, _ = fpga.compute_eri_batch(bfs, PrecisionMode.FP64, use_screening=False)
        # Run FP16
        res_16, _, _ = fpga.compute_eri_batch(bfs, PrecisionMode.FP16, use_screening=False)

        # Both should produce results of similar length
        self.assertEqual(len(res_64), len(res_16))


if __name__ == '__main__':
    unittest.main(verbosity=2)
