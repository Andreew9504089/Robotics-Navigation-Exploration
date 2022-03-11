# Homework 1 for Robotics Navigation and Exploration


## TODO

### Path Planning
- ☑️  A* planner
> *The planner couldn't find the exact goal, therefore an error of 20 is permitted.*
- [ ] RRT* planner
> *The planner behave like rrt, bugs still need to be fixed.*

### Basic Model
- ⚠️ LQR control <br>
> *Need to fix fast oscillation and will somtimes go off the track.*<br>
- ⚠️ PID control <br>
> *Not following the track in hw1 but act normally in lab1.*<br>
- ☑️ Pure Pursuit control <br>
- ☑️ Collision solution <br>

### Bicycle Model
- ⚠️ LQR control<br>
> 1. *Oscillation and went off the track at some point.*<br>
> 2. *Not following the track really well*<br>
- ⚠️ PID control<br>
> *Went off the track and stuck if the goal point is behind the vehicle.*<br>
- ☑️ Stanley control<br>
- ☑️ Pure Pursuit
- ☑️ Collision solution<br>

### Differential Model
😵 weird behaviors, might have problems in the transform from (v,w) to (lw,rw).<br>
- - [ ] LQR control
- - [ ] PID control
- - [ ] Stanley control
- - [ ] Collision solution
