FormulaProcesses
================

FormulaProcesses represent a single process.

During evaluation the FormulaModels are given the input variables to the process.
After the evaluation the result_variables are returned from the FormulaModels.

Graph propagation of input and result variables:
Req FP 0. FormulaProcesses can be constructed with direct input variables at initiation time
Req FP 1. FormulaProcesses can also define import variables that are imported from incoming edges. These can change depending on the result of other nodes evaluated earlier in the evaluation graph
Req FP 2. During variable import, input variables (those defined at process initialisation) are not overwritten by import variables
Req FP 3. direct input and result variables are stored with the process models
Req FP 4. imported variables from adjacent downstream processes in the graph are aggregated if they are aggregated variables



YAML Syntax
-----------
Service Models can be completely described in yaml files.
The yaml is made from a sequence of documents

- the formula text

Variables
=========
Req Var 0. Variables can be marked global if they are available to all processes in the service model

Framework
=========


Variable values:

processes have variables

at model evaluation, the processes will be assigned instance variables from the associated variables

these will be cached

values are


Export Variables
================
Upstream processes can import variables and downstream processes can export variables.

Import: Each process defines a list of import variables. The default is set to import `['data_volume', 'num_devices']`.
Only variables with names listed here will be imported. Additionally, no variable that is already defined in an upstream
process will be overridden by a variable exported by a downstream process.

Export: By default, a process exports all of its variables that are of type :class:`ExportVar`. Additional variables can be exported. Eg

```
    def export_variables(self, outgoing_edge, simulation_control):
        super().export_variables(outgoing_edge, simulation_control)
        if hasattr(self, 'bitrate'):
            data_volume = self.bitrate * self.service_on_time_per_day_mins * 60 * days_per_month * self.num_devices
            outgoing_edge[2]['data_volume'] = Variable.static_variable('data_volume', data_volume)  # b/s * s = b
```


Units
=====

At present, most variables are expected to be defined in Watt, Joules, bits, grams and combinations thereof.
The calculation takes place in the subclasses of :class:`ngmodel.ProcessModel` and specficially, the overridden methods of
:meth:`ngmodel.ProcessModel.energy_footprint` and :meth:`ngmodel.ProcessModel.embodied_carbon_footprint`.

Carbon footprint results from energy are converted to tonnes CO2e based on carbon intensity (see :meth:`ngmodel.ProcessModel.apply_carbon_intensity`).

Analysis Result Units
------------

After calculation, optional analysis takes place (see :ref:`Framework`). Results are stored in pandas Dataframes.
The load function used for these Dataframes also converts the results to MWh and tCO2e (see :funct:`util.convert`).