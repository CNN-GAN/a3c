-- action: vx, vy, vw, vl

step = 0.05
dx = step
dy = step
dw = step
dh = step/3.0
dl = step

collision_hd_1 = simGetCollectionHandle('robot_base')
collision_hd_2 = simGetCollectionHandle('obstacle_all')


function do_action(robot_hd, action)
    local current_pos=simGetObjectPosition(robot_hd,-1)
    local current_ori=simGetObjectQuaternion(robot_hd,-1)

    local sample_pose = {}
    sample_pose[1] = dx*action[1]
    sample_pose[2] = dy*action[2] 
    sample_pose[3] = dh*action[4] 

    local sample_ori = {}
    sample_ori[1] = current_ori[1] 
    sample_ori[2] = current_ori[2]   
    sample_ori[3] = current_ori[3] + dw*action[3] 
    sample_ori[4] = current_ori[4]

    -- print (sample_ori[3])
    simSetObjectPosition(robot_hd, robot_hd, sample_pose)
    simSetObjectQuaternion(robot_hd, -1, sample_ori)

    -- check collision --
    if is_valid() then 
        local final_pose = simGetObjectPosition(robot_hd,-1)
        return {final_pose[3]}, 't'
    else
        restore_pose(robot_hd, current_pos, current_ori)
        print ('collision')
        return {current_pos[3]}, 'f'      
    end
end

function restore_pose(robot_hd, poi, ori)
    simSetObjectPosition(robot_hd,-1,poi)
    simSetObjectQuaternion(robot_hd,-1,ori)
end

function is_valid()
    local is_valid = true
    local res=simCheckCollision(collision_hd_1, collision_hd_2)

    if res > 0 then 
        is_valid = false
    end

    return is_valid
end

