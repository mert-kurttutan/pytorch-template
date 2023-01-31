# Template for Structured Pytorch Project

This is created to start off pytorch project fast and efficiently by providing the skeleton code, for myself and others.

For presentation, I chose the CIFAR-10 problem since

* One of the simplest problems: making this repo more template-like without exposing problem specific complexity
* Not too simple: i.e. binary digit recognition is a real-life problem, despite being solved with current deep learning methods.


## Project Structure

* assets: items like picture, logo svg, etc
* bin: executables files
* configs: config files used for various settings, hyperparam setting, dataset setting, model setting, etc.
* docs: Any relevant documentation about the project
* notebooks: Jupyter notebooks for experimentation or tutorial
* scripts: python scripts to run training, validation etc.
* src: underlying source code


## Source Code

Here is the source code explanation



## Getting Started

For wandb logging, you need to set up a wandb account and your API_KEY for wandb service.


## TODO

- [ ] Multi GPU setting
- [ ] Option to resume training (respecting loggers)
- [ ] Add more logger option
- [ ] Maybe make this repo true template


## References
- Logger Interface: The idea of using Logger class as a wrapper for wandb is inspried by the ultralytics repo. This makes the logger utils more modular as we can put different logger platforms (e.g. clearml) and use the same Logger interface, which hides all the platform specifici details, at least for now.