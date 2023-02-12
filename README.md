# Template for Structured Pytorch Project

This is created to start off pytorch project fast and efficiently by providing the skeleton code, for myself and others.

For presentation, I chose the CIFAR-10 problem since

* One of the simplest problems: making this repo more template-like without exposing problem specific complexity
* Not too simple: i.e. image classificaton is a real-life problem, despite this particular example begin somewhat simple for the current status of deep learning

## WANDB Logs

You can see all the results (loss, accuracy, learning rate, pretrained models used) in the WANDB project page [here](https://wandb.ai/kurttutan-mert/pytorch-cifar10?workspace=user-). It also includes `%99.3 validation accuracy` on CIFAR10, very close to SOTA.

## Project Structure

* assets: items like picture, logo svg, etc
* bin: executables files
* configs: config files used for various settings, hyperparam setting, dataset setting, model setting, etc.
* docs: Any relevant documentation about the project
* notebooks: Jupyter notebooks for experimentation or tutorial
* scripts: any type of scripts to run, e.g. script for setting up virtual env.
* src: underlying source code


## Source Code

This section is important if you want to use this template in your workflow and or want to improve the template. The main feature of the project is to provide a structure as low level as possible while having important features, e.g. resume training, using checkpoints etc.

## Getting Started

For wandb logging, you need to set up a wandb account and your API_KEY for wandb service.

## TODO

* [X] Option to resume training (respecting loggers)
* [ ] Maybe make this repo true template
* [ ] Present other projects using this format
* [ ] Multi GPU setting

Update: I decided not to complete the last goal, I will defer this one to another project that focuses on open sourcing fine tuning benchmarks efficiently.

## Contributions

Any bug, issue, fix, feature is much appreciated.
Additionally, If you think that there is a component that is general enough to be put in this template, you are welcome to submit PR and why it should be added.

## References

* Logger Interface: The idea of using Logger class as a wrapper for wandb is inspried by the ultralytics repo. This makes the logger utils more modular as we can put different logger platforms (e.g. clearml) and use the same Logger interface, which hides all the platform specifici details (ideally).
