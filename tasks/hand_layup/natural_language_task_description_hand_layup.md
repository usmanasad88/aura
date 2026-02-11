Consider the dummy task video at /home/mani/Repos/aura/demo_data/layup_demo/layup_dummy_demo_crop_1080.mp4

In this task, the human is performing a hand layup on fiberglass on a mold. These are the objects of interest:
1. Fiberglass sheet (already cut in to appropriate sized pieces)
2. Metal Mold
3. Resin Bottle
4. Hardener Bottle
5. Cup with mixing stick
6. Weigh scale
7. Brush (Small and Medium)
8. Roller

A frame with all the required objects laid out on the table is at /home/mani/Repos/aura/demo_data/layup_demo/layup_dummy_frame.jpg

Here's the general process: Put cup on weigh scale. Add resin. Add hardener. Weigh the mixture. Fix the mixture. Put layer 1 on mold. Add resin to first layer with brush. Add layer 2 of fiberglass. Add resin to second layer with brush. Add layer 3 of fiberglass. Add resin to layer 3 with brush. Add layer 4 of fiberglass. Add resin to layer 4 with brush. Use roller to consolidate. The robot can help by placing objects on the table when required and house-keeping, i.e. moving objects from the table to the storage (on the second table) when no longer needed. The system is also supposed to closely monitor the process. For example, monitor that the human is wearing gloves. The current ratio of resin/hardener is added. The current ratio of resin/fiber is used. The shelf life on the mixture is not exceeded. This is a dummy video where the human operator is essentially pretending. 

We have to follow the process outlined in the draft paper, and partially demonstrated in the previous example task of bottle weighing. 

Create the scripts and config files for this task. First run the script on the dummy video, in which the robot is not active. Create a state config, the task graph, the robot affordance monitor (robot can only move objects from the storage to the workplace) Monitor the state, completed task, in-progress task, next expected task. SAM3 can be used to monitor objects of interest in the perception module. Create robot instructions (which will not be executed in the dummy video). For example, when an object is no longer to be used again, the robot command to remove it from the workplace and shift to stage can be issued by the "brain". 

Create a visualization which overlays important outputs of the modules on the input /home/mani/Repos/aura/demo_data/layup_demo/layup_dummy_demo_crop_1080.mp4 video