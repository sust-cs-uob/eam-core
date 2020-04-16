import unittest

from eam_core import SimulationControl, ServiceModel, np, Formula, FormulaModel, FormulaProcess, generate_static_variable


class TestDSLModels(unittest.TestCase):

    def test_formula_process(self):
        test_formula = """
                energy = energy_intensity * data_volume
                """
        sim_control = SimulationControl()
        sim_control.sample_size = 1

        process_variables = {
            'energy_intensity': generate_static_variable(sim_control, 'energy_intensity', 5, random=False)}

        fmodel = FormulaModel(Formula(test_formula))

        result_variables = {'energy': 'test_energy'}
        p = FormulaProcess('test', fmodel, process_variables, result_variables)

        ingress_variables = {'data_volume': generate_static_variable(sim_control, 'data_volume', 5, random=False)}
        result_variables = p.evaluate(sim_control, ingress_variables)
        assert 'test_energy' in result_variables
        # print(export_variable_names['test_energy'].data_source.get_value('', sim_control))
        assert result_variables['test_energy'].data_source.get_value('', sim_control) == 25

    # @todo review return values from DSL
    @unittest.skip("old DSL. New does not return values - need review")
    def test_basic_formula_model(self):
        """
        Test a model with a dsl process
        :return:
        """
        sim_control = SimulationControl()
        sim_control.sample_size = 1

        s = ServiceModel()

        input_variables = {
            'data_volume_var': generate_static_variable(sim_control, 'data_volume_var', 5, random=False)}
        dvpm = FormulaModel(Formula("data_volume = data_volume_var"), )
        # todo - define return variables
        export_variables = {'data_volume': 'data_volume'}
        dvp = FormulaProcess('aux process', dvpm, input_variables=input_variables,
                             export_variable_names=export_variables)

        test_formula = """
        energy = energy_intensity * data_volume
        return energy
        """

        input_variables = {
            'energy_intensity': generate_static_variable(sim_control, 'energy_intensity', 5, random=False)}

        fmodel = FormulaModel(Formula(test_formula), )

        import_variables = {'data_volume': {'aggregate': True}}
        p = FormulaProcess('test', fmodel, input_variables=input_variables, import_variable_names=import_variables)

        s.link_processes(dvp, p)
        fp = s.footprint(embodied=False, simulation_control=sim_control)

        assert np.allclose(fp['use_phase_energy']['test'], [25.], rtol=.1)
