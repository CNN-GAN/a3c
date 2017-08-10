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

function step(inInts,inFloats,inStrings,inBuffer)
    -- print (#inFloats)
    robot_state, res = do_action(robot_hd, joint_hds, inFloats, start_joints)
    -- sample_obstacle_position()

    return {}, robot_state, {}, res
end


function get_global_path(inInts,inFloats,inStrings,inBuffer)
    -- return {},{},{},'f'
    path_in_robot_frame = transform_path_to_robotf(path_dummy_list, robot_hd)

    -- path_in_robot_frame = {}
    -- local d_hd = target_hd
    -- local d_pos = simGetObjectPosition(d_hd, robot_hd)

    -- -- local dist = math.sqrt(d_pos[1]*d_pos[1] + d_pos[2]*d_pos[2])
    -- -- local angle_thigh = math.atan(d_pos[2]/d_pos[1])

    -- -- path_in_robotf[#path_in_robotf + 1] = dist
    -- -- path_in_robotf[#path_in_robotf + 1] = angle_thigh
    -- path_in_robot_frame[#path_in_robot_frame + 1] = d_pos[1]
    -- path_in_robot_frame[#path_in_robot_frame + 1] = d_pos[2]

    -- print('path length: '..#path_in_robot_frame)
    -- print(#path_in_robot_frame)
    return {}, path_in_robot_frame, {}, ''

end

function generate_path()
    init_params(1, 0, 'robot_base', 'obstacle_tall', false, false)
    task_hd, state_dim = init_task('base_yaw','task_1')
    path = compute_path(task_hd, 10)
    -- displayInfo('finish 1 '..#path)
    -- applyPath(task_hd, path, 0.1)
    simExtOMPL_destroyTask(task_hd)

    return path
end

function applyPath(task_hd, path, speed)
    local state = {}
    for i=1,#path-state_dim,state_dim do
        for j=1,state_dim,1 do
            state[j]=path[i+j-1]
        end
        res = simExtOMPL_writeState(task_hd, state) -- 12 joints, yaw,x,y

        sleep(speed)
        simSwitchThread()
    end
end

function create_dummy(pos)
    local hd = simCreateDummy(0.1)
    pos[3] = pos[3]+0.2
    simSetObjectPosition(hd, -1, pos)
    -- simSetObjectQuaternion(hd, -1, ori)
    -- path_dummy_list[#path_dummy_list+1] = hd
    return hd
end

function remove_dummy()
    for i=1, #path_dummy_list, 1 do
        local object_hd = path_dummy_list[i]
        res = simRemoveObject(object_hd)
    end 
end

function create_path_dummy(path)
    local dummy_list = {}
    local pos={}
    for i=1, #path, 2 do  
        pos[1] = path[i]
        pos[2] = path[i+1]
        pos[3] = 0
        local hd = create_dummy(pos)
        dummy_list[#dummy_list+1] = hd
    end
    return dummy_list
end

function transform_path_to_robotf(path_d_list, robot_hd)
    path_in_robotf = {}
    for i=1, #path_d_list, 1 do
        local d_hd = path_d_list[i]
        local d_pos = simGetObjectPosition(d_hd, robot_hd)

        local dist = math.sqrt(d_pos[1]*d_pos[1] + d_pos[2]*d_pos[2])
        local angle = math.atan(d_pos[2]/d_pos[1])

        -- path_in_robotf[#path_in_robotf + 1] = dist
        -- path_in_robotf[#path_in_robotf + 1] = angle

        path_in_robotf[#path_in_robotf + 1] = d_pos[1]
        path_in_robotf[#path_in_robotf + 1] = d_pos[2]
    end

    return path_in_robotf
end

function sample_obstacle_position(obs_hds, num)
    local v = 0.02

    for i=1, num, 1 do
        obs_pos = simGetObjectPosition(obs_hds[i], -1)
        obs_pos[1] = (math.random()-0.5) * 0.1 + obs_pos[1]
        obs_pos[2] = (math.random()-0.5) * 0.1 + obs_pos[2]

        if obs_pos[1] > 2.5 then
            obs_pos[1] = 2.5
        elseif obs_pos[1] < -2.5 then 
            obs_pos[1] = -2.5
        end

        if obs_pos[2] > 1 then
            obs_pos[2] = 1
        elseif obs_pos[2] < -1 then 
            obs_pos[2] = -1
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

    if level < 0 then
        -- set robot --
        simSetObjectPosition(robot_hd, -1, pre_pos)
        simSetObjectPosition(fake_robot_hd, -1, pre_pos)

        simSetObjectQuaternion(robot_hd, -1, pre_ori)
        simSetObjectQuaternion(fake_robot_hd, -1, pre_ori)

        set_joint_positions(joint_hds, start_joints)

        -- set target --
        simSetObjectPosition(target_hd, -1, pre_tar_pose)
        return 0
    end

    print ('level: '..level)
    local max_r = 3
    if level == 10 then      -- 1 meter around the robot 
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

    elseif level == 1 then      -- 1 meter around the robot 
        -- sample initial robot pose
        robot_pos[1] = (math.random() - 0.5) * 2
        robot_pos[2] = (math.random() - 0.5) * 2
        robot_pos[3] = start_pos[3]

        robot_ori[3] = (math.random() - 0.5) * math.pi

        -- sample initial target pose
        target_pos[1] = (math.random() - 0.5) * 2 + robot_pos[1]
        target_pos[2] = (math.random() - 0.5) * 2 + robot_pos[2]
        target_pos[3] = 0

    elseif level == 2 then      -- 2 meter around the robot
        -- sample initial robot pose
        robot_pos[1] = (math.random() - 0.5) * max_r
        robot_pos[2] = (math.random() - 0.5) * max_r
        robot_pos[3] = start_pos[3]

        robot_ori[3] = (math.random() - 0.5) * math.pi

        -- sample initial target pose
        target_pos[1] = (math.random() - 0.5) * max_r
        target_pos[2] = (math.random() - 0.5) * max_r
        target_pos[3] = 0

    elseif level == 3 then      -- 2 meter around the robot with few osbtacles
        -- sample initial robot pose
        robot_pos[1] = (math.random() - 0.5) * max_r
        robot_pos[2] = (math.random() - 0.5) * max_r
        robot_pos[3] = start_pos[3]

        robot_ori[3] = (math.random() - 0.5) * math.pi

        -- sample initial target pose
        target_pos[1] = (math.random() - 0.5) * max_r
        target_pos[2] = (math.random() - 0.5) * max_r
        target_pos[3] = 0

        sample_obstacle_position(obs_hds, #obs_hds/2)
    else        -- 2 meter around the robot with random obstacle every time
        -- sample initial robot pose
        robot_pos[1] = (math.random() - 0.5) * max_r
        robot_pos[2] = - math.random() - 1
        robot_pos[3] = start_pos[3]

        robot_ori[3] = (math.random() - 0.5) * math.pi

        -- sample initial target pose
        target_pos[1] = (math.random() - 0.5) * max_r
        target_pos[2] = math.random() + 1
        target_pos[3] = 0
    
        sample_obstacle_position(obs_hds, #obs_hds)
    end

    -- set robot --
    simSetObjectPosition(robot_hd,-1,robot_pos)
    simSetObjectPosition(fake_robot_hd,-1,robot_pos)

    simSetObjectQuaternion(robot_hd,-1,robot_ori)
    simSetObjectQuaternion(fake_robot_hd,-1,robot_ori)

    set_joint_positions(joint_hds, start_joints)

    -- set target --
    simSetObjectPosition(target_hd,-1,target_pos)

    -- check collision for robot pose --
    local res_robot = simCheckCollision(robot_body_hd, obstacle_all_hd)
    local res_target = simCheckCollision(target_hd, obstacle_all_hd)

    -- print (res_robot, res_target)
    pre_pos = robot_pos
    pre_ori = robot_ori
    pre_tar_pose = target_pos 

    return res_robot+res_target
    -- print (res_robot, res_target)
    -- g_path = generate_path()
    -- path_dummy_list = create_path_dummy(g_path)
end

function init(level)
    remove_dummy()
    local init_value = 1
    while (init_value ~= 0) do
        init_value = sample_init(level)
    end

    scale = scale + 0.0001
    if scale > 1 then
        scale = 1
    end 

    g_path = generate_path()
    path_dummy_list = create_path_dummy(g_path)

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



