package.path=package.path .. ";/home/xi/workspace/a3c_palnner/environment/?.lua"
require("common_functions")
require("ompl_functions")
require("robot_control")

-- simSetThreadSwitchTiming(2) 
-- simExtRemoteApiStart(19999)

-------- remote functions ---------------------
function reset(inInts,inFloats,inStrings,inBuffer)
    print ('reset !!')
    local level = inFloats[1]
    print (level)
    init(level)
    return {}, {}, {}, ''
end

function step_hlc(inInts,inFloats,inStrings,inBuffer)

    local robot_pos=simGetObjectPosition(robot_hd, -1)
    simSetObjectPosition(fake_robot_hd, -1, robot_pos)

    res = move_hlc(fake_robot_hd, inFloats)
    -- sample_obstacle_position()

    return {}, {}, {}, res
end

function get_dist(inInts,inFloats,inStrings,inBuffer)
    -- print (#inFloats)
    local robot_pos=simGetObjectPosition(robot_hd, fake_robot_hd)
    local dist = math.sqrt(robot_pos[1]*robot_pos[1] + robot_pos[2]*robot_pos[2])

    return {}, {dist}, {}, res
end

function step(inInts,inFloats,inStrings,inBuffer)
    -- print (#inFloats)
    robot_state, res = do_action(robot_hd, joint_hds, inFloats, start_joints)
    -- sample_obstacle_position()

    return {}, robot_state, {}, res
end


function sample_obstacle_position(obs_hds, num)
    local v = 0.02

    for i=1, num, 1 do
        obs_pos = simGetObjectPosition(obs_hds[i], -1)
        obs_pos[1] = (math.random()-0.5) * 4
        obs_pos[2] = (math.random()-0.5) * 3

        if obs_pos[1] > 2.5 then
            obs_pos[1] = 2.5
        elseif obs_pos[1] < -2.5 then 
            obs_pos[1] = -2.5
        end

        if obs_pos[2] > 2.5 then
            obs_pos[2] = 2.5
        elseif obs_pos[2] < -2.5 then 
            obs_pos[2] = -2.5
        end
        print(obs_pos[1], obs_pos[2])
        simSetObjectPosition(obs_hds[i], -1, obs_pos)
    end
end

function sample_init(level)
    local robot_pos = {}
    local target_pos = {}
    local robot_ori = start_ori

    set_joint_positions(joint_hds, start_joints)

    -- sample initial robot pose
    print ('in level: '..level)
    robot_pos[1] = 0
    robot_pos[2] = 0
    robot_pos[3] = start_pos[3]

    robot_ori[3] = 0

    -- sample initial target pose
    target_pos[1] = 3
    target_pos[2] = 0
    target_pos[3] = 0

    -- sample_obstacle_position(obs_hds, #obs_hds)

    -- set robot --
    simSetObjectPosition(robot_hd,-1,robot_pos)
    simSetObjectPosition(fake_robot_hd,-1,robot_pos)

    simSetObjectQuaternion(robot_hd,-1,robot_ori)
    simSetObjectQuaternion(fake_robot_hd,-1,robot_ori)

    set_joint_positions(joint_hds, start_joints)

    -- check collision for robot pose --
    local res_robot = simCheckCollision(robot_body_hd, obstacle_all_hd)

    -- print (res_robot, res_target)
    pre_pos = robot_pos
    pre_ori = robot_ori

    return res_robot
end

function init(level)
    local init_value = 1
    while (init_value ~= 0) do
        init_value = sample_init(level)
    end

    print ('init!')
    return 1
end

g_path = {}
path_in_robot_frame = {}
path_dummy_list = {}

start_joints = {}

x_range = 2
y_range = 2

scale = 0.05

robot_body_hd = simGetCollectionHandle('robot_body')
obstacle_all_hd = simGetCollectionHandle('obstacle_all')
obstacle_low_hd = simGetCollectionHandle('obstacle_low')
obs_hds = simGetCollectionObjects(obstacle_low_hd)


target_hd = simGetObjectHandle('target')
robot_hd = simGetObjectHandle('rwRobot')
fake_robot_hd = simGetObjectHandle('base_yaw')
joint_hds = get_joint_hds()

start_pos = simGetObjectPosition(robot_hd, -1)
start_joints = get_joint_positions(joint_hds)
start_ori = simGetObjectQuaternion(robot_hd,-1)

pre_pos = start_pos
pre_ori = start_ori
pre_tar_pose = start_pos 

-- init()
-- sleep(2)

-- init()
-- g_path = generate_path()
-- path_dummy_list = create_path_dummy(g_path)

-- action = {1, 1, 0, -1, -1}
-- act = do_action(robot_hd, joint_hds, action)
-- print (act[1], act[2])

while simGetSimulationState()~=sim_simulation_advancing_abouttostop do
    -- do something in here
    simSwitchThread()
end



--simExtOMPL_destroyTask(task_hd)

--init_params(6, 12, 'robot_body', 'obstacles', false, true)
--task_hd2, state_dim = init_task('start','task_2')
--path_2 = compute_path(task_hd2, 50)
--applyPath(task_hd2, path_2, 0.2)

--displayInfo('finish 2 '..#path..' '..#path_2)

-- while true do
--     sleep(0.01)
--     simSwitchThread()
-- end



