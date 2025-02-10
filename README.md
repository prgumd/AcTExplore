<h1 align="center">
  AcTExplore
</h1>
<h2 align="center">
  Active Tactile Exploration on Unknown Objects
</h2>

<div align="center">
  <a href="https://amirshahid.github.io/">Amir-Hossein Shahidzadeh</a> &nbsp;•&nbsp;
  <a href="https://sj-yoo.info/">Seong Jong Yoo</a> &nbsp;•&nbsp;
  <a href="https://www.cs.umd.edu/people/mppavan">Pavan Mantripragada</a> &nbsp;•&nbsp;
  <a href="https://chahatdeep.github.io/">Chahat Deep Singh</a> &nbsp;•&nbsp;
  <a href="http://www.cfar.umd.edu/~fer/">Cornelia Ferm&#252ller</a> &nbsp;•&nbsp;
  <a href="http://www.cfar.umd.edu/~yiannis/">Yiannis Aloimonos</a>
  <br/>
  IEEE International Conference on
Robotics and Automation (<a href="https://2024.ieee-icra.org/">ICRA</a>) 2024
</div>

<h4 align="center">
  <a href="https://prg.cs.umd.edu/AcTExplore"><b>Website</b></a> &nbsp;•&nbsp;
  <a href="https://arxiv.org/abs/2310.08745"><b>Paper</b></a> &nbsp;•&nbsp; 
  <a href="https://youtu.be/38utg590wao"><b>YouTube</b></a>
</h4>

<div align="center">
<em><b>TL;DR</b>: We explore and reconstruct the object's surface with robotic fingers equipped with tactile sensor</em>
<br> <br>
</div>

<div align="center">
  <img src="https://prg.cs.umd.edu/research/AcTExplore_files/overview.gif"
  width="80%">
</div>



## Content
This package contains several components:


## Installation:
- Clone repo
- python==3.9
- Install pytorch >= 2.2.1
- pip install -r requirements.txt
  

- For Training
	- Generate 'Training/Logs' folder
	- Set configuration at *conf/RL.yaml*
		- state
			- input_type: TTA, TTS, depth
		- reward
			- type: TM, AM, AMB
	- Execute `python PPO.py` ....
- For testing
	- Generate 'outputs' folder
	- Set configuration at *conf/test.yaml*
		- RL
			- algorithm: PPO, RecurrentPPO
			- pretrain_model_path: 'best/model/path'
		- state
			- input_type: TTA, TTS, depth (has to match with pretrain_model settings)
		- reward
			- type: TM, AM, AMB (has to match with pretrain_model settings)
		- environment
			- object
				- object_name: test object name
				- urdf_path: `["objects/ycb/object/model.urdf"]`
	- Execute `python test.py`

## Exploration in Tacto
<div align="center">
  <img src="./sim.gif"
  width="80%">
</div>


## Reconstruction
<div align="center">
  <img src="https://prg.cs.umd.edu/research/AcTExplore_files/reconstruction.gif"
  width="80%">
</div>

## Bibtex

```
@misc{shahidzadeh2023actexplore,
      title={AcTExplore: Active Tactile Exploration on Unknown Objects}, 
      author={Amir-Hossein Shahidzadeh and Seong Jong Yoo and Pavan Mantripragada and Chahat Deep Singh and Cornelia Fermüller and Yiannis Aloimonos},
      year={2023},
      eprint={2310.08745},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

