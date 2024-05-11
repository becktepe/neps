!!! note "Work in Progress"
    This document is currently a work in progress and may contain incomplete or preliminary information.

## Introduction
### Configuring with YAML
Configure your experiments using a YAML file, which serves as a central reference for setting up your project.
This approach simplifies sharing, reproducing, and modifying configurations.
#### Simple YAML Example
Below is a straightforward YAML configuration example for NePS covering the required arguments.
=== "config.yaml"
    ```yaml
    --8<-- "tests/test_yaml_run_args/test_declarative_usage_docs/simple_example.yaml"
    ```

=== "run_neps.py"
    ```python
    import neps
    def run_pipeline(learning_rate, optimizer, epochs):
        pass
    neps.run(run_pipeline, run_args="path/to/your/config.yaml")
    ```


#### Advanced Configuration with External Pipeline
In addition to setting experimental parameters via YAML, this configuration example also specifies the pipeline function
and its location, enabling more flexible project structures.
=== "config.yaml"
    ```yaml
    --8<-- "tests/test_yaml_run_args/test_declarative_usage_docs/simple_example_including_run_pipeline.yaml"
    ```
=== "run_neps.py"
    ```python
    import neps
    # No need to define run_pipeline here. NePS loads it directly from the specified path.
    neps.run(run_args="path/to/your/config.yaml")
    ```

#### Extended Configuration
This example showcases a more comprehensive YAML configuration, which includes not only the essential parameters
but also advanced settings for more complex setups:
=== "config.yaml"
    ```yaml
    --8<-- "tests/test_yaml_run_args/test_declarative_usage_docs/full_configuration_template.yaml"
    ```

=== "run_neps.py"
    ```python
    import neps
    # Executes the configuration specified in your YAML file
    neps.run(run_args="path/to/your/config.yaml")
    ```

The `searcher` key used in the YAML configuration corresponds to the same keys used for selecting an optimizer directly
through `neps.run`. For a detailed list of integrated optimizers, see [here](optimizers.md#list-available-searchers)
!!! note "Note on Undefined Keys"
    Not all configurations are explicitly defined in this template. Any undefined key in the YAML file is mapped to
    the internal default settings of NePS. This ensures that your experiments can run even if certain parameters are
    omitted.

## Different Use Cases
### Customizing neps optimizer
```yaml
--8<-- "tests/test_yaml_run_args/test_declarative_usage_docs/customizing_neps_optimizer.yaml"
```
TODO
information where to find parameters of included searcher, where to find optimizers names...link


### Load your own optimizer
```yaml
--8<-- "tests/test_yaml_run_args/test_declarative_usage_docs/loading_own_optimizer.yaml"
```
### How to define hooks?

```yaml
--8<-- "tests/test_yaml_run_args/test_declarative_usage_docs/defining_hooks.yaml"
```
### What if your search space is big?
```yaml
--8<-- "tests/test_yaml_run_args/test_declarative_usage_docs/outsourcing_pipeline_space.yaml"
```

pipeline_space.yaml
```yaml
--8<-- "tests/test_yaml_run_args/test_declarative_usage_docs/pipeline_space.yaml"
```

### If you experimenting a lot with different optimizer settings
```yaml
--8<-- "tests/test_yaml_run_args/test_declarative_usage_docs/outsourcing_optimizer.yaml"
```

searcher_setup.yaml:
```yaml
--8<-- "tests/test_yaml_run_args/test_declarative_usage_docs/set_up_optimizer.yaml"
```

### Architecture search space (Loading Dict)
```yaml
--8<-- "tests/test_yaml_run_args/test_declarative_usage_docs/loading_pipeline_space_dict.yaml"
```
search_space.py
```python
search_space = {}
```

