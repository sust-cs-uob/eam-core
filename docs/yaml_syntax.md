# Overview
The yaml files specify the processes that form a tree as part of a model.

The two required segments are `Processes` and `Metadata`. The `Metadata` contains the `model_name`.

Here is a simple example:
```yaml
Processes:
  - name: process a
    formula:
      text: |
       return a + b + c
    input_variables:
      - formula_name: a
        type: StaticVariable
        value: 6
    import_variables:
      - b
      - c
  - name: process b
    formula:
      text: |
        c = 3
        b = 2
    export_variables:
      b: b
      c: c
    link_to:
      - process a
Metadata:
  model_name: test
```

This model has two processes `process a` and `process b`.

# Processes
Each process has a
`name` and a `formula` and declares or defines variables that are used in the `formula`.

## Formula
The formula has either text as in
```
formula:
      text: |
       return a + b + c
```
or it contains a reference to a formula (below).

A formula just like your ordinary maths formulas with variables, simple arithmetic (+,-,/,*).
A formula can have `return` statement. In this case, the value returned by this formula will be included in the
overall model result.

### Variables
Variables are used in a formula are defined in a section `input_variables` and `import_variables`.

#### `import_variables`
It is possible for a process to use variables in a formula that come from another linked, upstream process.
In order for this to work a process must define `export_variables`

Example:
```yaml
Processes:
  - name: process a
    formula:
      text: |
       return 6 + b + c
    import_variables:
      - b
      - c
  - name: process b
    formula:
      text: |
        b = 2
        c = 3
    export_variables:
      - b
      - c
    link_to:
      - process a
Metadata:
  model_name: test
```
The above model defines two processes `process a` and `process b`. `process b` defines variables `b` and `c` and has a
section `export_variables` that lists those variables.

#### `export_variables` renaming
It is possible to export a variable under a different name.
In the below statement, the variables `b` and `c` will be published under these names.
```
export_variables:
  - b
  - c
```

Alternatively, in the below `b` will be published as `d` and `c` as `c`.
```
export_variables:      
  b: d
  c: c
```

In other words, the sequence in export_variable dicts is 'model var name':'global var name'


Each `export_variables` section must use either use the explicit or implicit syntax for all variables.


#### `import_variables` renaming
Similar to `export_variables` renaming, it is possible to rename import variables.

Without renaming:
```yaml
import_variables:
  - b
  - c
```

With renaming:
```
import_variables:
  - formula_name: c
    external_name: b
    aggregate: true
```
Note, how the import variables syntax is more explicit. It also allows setting aggregation in case several incoming 
edges include the same variable.

#### `input_variables`

For example:
```
formula:
  text: |
   return a + b + c
input_variables:
  - formula_name: a
    type: StaticVariable
    value: 6

```
Here, variable `a` has the value 6.

There are two ways to define individual variables: `StaticVariable` or `ExcelVariable`.

Example:
```yaml
- formula_name: custom_coefficient
  type: ExcelVariable
  file_alias: test_location
  sheet_name: Sheet1
  excel_var_name: a
```
The `file_alias` must be matched in the `file_locations` section in the `Metadata`.

There are correspondingly two ways to define several StaticVariables and ExcelVariables:

```yaml
- type: StaticVariables
  variables:
    passive_standby_time_per_day_mins: 0
    service_active_standby_time_per_day_mins: 0
    total_active_standby_time_per_day_mins: 0
    total_on_time_per_day_mins: 1440

- type: ExcelVariableSet
  file_alias: public_model_params
  sheet_name: UserDeviceInputVars
  substitution: dtt
  variables:
    - power: Power
    - bitrate
    - power_active_standby
    - power_passive_standby
```


### Metadata `file_locations`

There are two types: `google_spreadsheet` which are downloaded from the cloud or direct file references.

Example:
```yaml
Metadata:
  model_name: test model
  file_locations:
    - file_alias: test_location
      file_name: data/test_data.xlsx
      type: google_spreadsheet
      google_id_alias: scenarios_public_params
    - file_alias: dummy_location
      file_name: _tmp_data/dummy.xlsx
```

The `google_id_alias` will be used to lookup the id in the project config file `google_drive_params_sheet_id_alias` parameter
which is a dictionary string that provides the spreadsheet id for each `google_id_alias` value.
The `file_name` is the location where the google spreadsheet will be locally cached at.

### `link_to`
A process can be linked to other nodes by listing all the names of the names in the `link_to` section

```
link_to:
 - a
 - b
```

### Formula references
The main YAML file can contain a top-level section `Formulas` that defines formulas with a name. These can be refered
to by processes.

```yaml

Formulas:
  - name: basic intensity-volume formula
    text: |
      energy = energy_intensity * data_volume
      return energy
Processes:
  - name: process a
    formula:
      ref: basic intensity-volume formula
```

## Formula Text Inline Tests

It is possible to define tests with formulas. Example below.

The `test_config` includes the required configuration of all variables.
Variables listed in the `variables` dictionary will be set to their respective values.
All variables not explicitly listed will be set to 1.

```yaml
text: |
  average_bitrate = (share_videos * video_bitrate + share_audio_streams * audio_bitrate)
  data_volume = video_minutes_monthly * 60 * 1e6 * average_bitrate * traffic_flow_coefficient
  energy = energy_intensity * data_volume
  return energy
test_config:
  variables:
    share_videos: 2
    share_audio_streams: 2
    traffic_flow_coefficient: 1/1e6
    video_minutes_monthly: 1/60
    energy_intensity: 2
  expect: 8
```

## Constants
It is possible to define constants to be used as global variables
```yaml
Constants:
  - type: StaticVariables
    variables:
      J_p_kWh: 3.60E+06
      months_per_year: 12
      days_per_month: 30.41
```
And then use them like so
```yaml
  - name: Playout
    formula:
      text: |
        return playout_annual_energy * J_p_kWh / months_per_year
    input_variables:
      - type: Constants
        variables:
          - J_p_kWh
          - months_per_year
```

## Simulation Control
The metadata can contain the following elements to control the Simulation:
```yaml
  start_date: 2016-01-01
  end_date: 2032-01-01
  sample_size: 2
  sample_mean: False
```

In order to disable time series calculations, simply comments out ('#') the start and end date.

## Prototype Processes
Prototype provide default configurations that processes can add/ overwrite specific parameters of.

```yaml
- name: proto
        type: prototype
        metadata:
          device_type: User Device
        formula:
          text: |
           return a         

      - name: a
        prototype: proto
        formula:
          text: |
           return a + b
        input_variables:
          - formula_name: a
            type: StaticVariable
            value: 6
        import_variables:
          - b       
``` 
In the example a prototype process is created and used by the 'process a'.

# Analysis section

## Example

```yaml
Analysis:
  # these are stored with uncertainty
  result_variables:
  - energy
  - carbon

  # for these excel tables are created
  numerical:
  - carbon
  - energy

  # the scenarios will be evaluated. sub directories in the output folder contain the output
  scenarios:
    - AO_10
    - AO_25
    - AO_50

  # declaration of plots to create
  plots:
    - name: device_groups
      variable: energy
      kind: bar
      groups:
        - name: User Devices
          categories:
            device_type: User Device
        - name: Residential Access Network
          categories:
            device_type: Wired Access Network
        - name: Core and Metro Network
          categories:
            platform_name : Core Network
        - name: YouTube Servers
          categories:
            platform_name: Google

  # unit conversions
  units:
    # includes idle_energy, on_energy
    - endswith: energy
      to_unit: GWh
    - __eq__: carbon
      to_unit: Mt
    - __eq__: data_volume
      to_unit: TB
    - __eq__: time
      to_unit: kyear
    - startswith: video_minutes_monthly
      to_unit: kyear
    - __eq__: result
      to_unit: GWh

  # graphviz process tree process node decorations
  process_group_colours:
    # group by this process metadata attribute 
    category_name: platform_name
    # apply these colours to groups
    colours:
      DSL: #037E8C
      Cellular: #F26C27
      Google: #024959
      Fibre: #024959
      Cable: #153A52
```