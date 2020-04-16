# Reference Time Frames

Energy related models always have a ref time. Our preferred solution is to explicitly scale processes to a ref duration 
inside models.

Eg


```yaml
- name: Playout
    formula:
      text: |
        aggreate_power = mean_power_per_linear_channel * number_of_BBC_linear_channels

        energy = aggreate_power * ref_duration

        return energy

```


# Debugging support / techniques 

Set log level to 'DEBUG' in logconf.yml