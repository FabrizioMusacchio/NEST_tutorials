# NEST Simulator turorials

This is a collection of tutorials showing how to use the NEST Simulator. You can find a detailed description for each tutorial in the following blog posts:

* [NEST simulator – A powerful tool for simulating large-scale spiking neural networks](https://www.fabriziomusacchio.com/blog/2024-06-09-nest_SNN_simulator/)
* [Step-by-step NEST single neuron simulation](https://www.fabriziomusacchio.com/blog/2024-06-16-nest_single_neuron_example/)


For reproducibility:

```bash
conda create -n nest -y python=3.11
conda activate nest
conda install -c conda-forge -y mamba
mamba install -y ipykernel matplotlib numpy pandas nest-simulator
```


![img](thumb/single_neuron_iaf_psc_alpha_Ie_376.0.png)