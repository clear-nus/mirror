import numpy as np
import carla
import pygame


def get_bounding_boxes(objects, camera, type):
    """
    Creates 3D bounding boxes based on carla vehicle list and camera.
    """
    if type == 'wall':
        bounding_boxes = []
        for object in objects:
            bounding_boxes.append(get_bounding_box(object, camera, type))
    else:
        bounding_boxes = [get_bounding_box(object, camera, type) for object in objects]
    # filter objects behind camera
    bounding_boxes = [bb for bb in bounding_boxes if all(bb[:, 2] > 0)]
    return bounding_boxes

def draw_bounding_boxes(display, bounding_boxes, type):
    """
    Draws bounding boxes on pygame display.
    """
    if type == 'wall':
        BB_COLOR = (0, 0, 255)
    else:
        BB_COLOR = (248, 64, 24)
    bb_surface = pygame.Surface((1280, 720))
    bb_surface.set_colorkey((0, 0, 0))
    for bbox in bounding_boxes:
        points = [(int(bbox[i, 0]), int(bbox[i, 1])) for i in range(8)]
        # draw lines
        # base
        pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
        pygame.draw.line(bb_surface, BB_COLOR, points[0], points[1])
        pygame.draw.line(bb_surface, BB_COLOR, points[1], points[2])
        pygame.draw.line(bb_surface, BB_COLOR, points[2], points[3])
        pygame.draw.line(bb_surface, BB_COLOR, points[3], points[0])
        # top
        pygame.draw.line(bb_surface, BB_COLOR, points[4], points[5])
        pygame.draw.line(bb_surface, BB_COLOR, points[5], points[6])
        pygame.draw.line(bb_surface, BB_COLOR, points[6], points[7])
        pygame.draw.line(bb_surface, BB_COLOR, points[7], points[4])
        # base-top
        pygame.draw.line(bb_surface, BB_COLOR, points[0], points[4])
        pygame.draw.line(bb_surface, BB_COLOR, points[1], points[5])
        pygame.draw.line(bb_surface, BB_COLOR, points[2], points[6])
        pygame.draw.line(bb_surface, BB_COLOR, points[3], points[7])
    display.blit(bb_surface, (0, 0))

def get_bounding_box(object, camera, type):
    """
    Returns 3D bounding box for a vehicle based on camera view.
    """
    bb_cords = create_bb_points(object)
    cords_x_y_z = object_to_sensor(bb_cords, object, camera, type)[:3, :]
    cords_y_minus_z_x = np.concatenate([cords_x_y_z[1, :], -cords_x_y_z[2, :], cords_x_y_z[0, :]])
    bbox = np.transpose(np.dot(camera.calibration, cords_y_minus_z_x))
    camera_bbox = np.concatenate([bbox[:, 0] / bbox[:, 2], bbox[:, 1] / bbox[:, 2], bbox[:, 2]], axis=1)
    return camera_bbox

def create_bb_points(object):
    """
    Returns 3D bounding box for an object.
    """
    cords = np.zeros((8, 4))
    extent = object.bounding_box.extent
    cords[0, :] = np.array([extent.x, extent.y, -extent.z, 1])
    cords[1, :] = np.array([-extent.x, extent.y, -extent.z, 1])
    cords[2, :] = np.array([-extent.x, -extent.y, -extent.z, 1])
    cords[3, :] = np.array([extent.x, -extent.y, -extent.z, 1])
    cords[4, :] = np.array([extent.x, extent.y, extent.z, 1])
    cords[5, :] = np.array([-extent.x, extent.y, extent.z, 1])
    cords[6, :] = np.array([-extent.x, -extent.y, extent.z, 1])
    cords[7, :] = np.array([extent.x, -extent.y, extent.z, 1])
    return cords

def object_to_sensor(cords, object, sensor, type):
        """
        Transforms coordinates of a vehicle bounding box to sensor.
        """

        world_cord = object_to_world(cords, object, type)
        sensor_cord = world_to_sensor(world_cord, sensor)
        return sensor_cord

def object_to_world(cords, object, type):
    """
    Transforms coordinates of an object bounding box to world.
    """
    if type == 'wall':
        object_loc = object.bounding_box.location
        bb_transform = carla.Transform(carla.Location(x=object_loc.x - object.transform.location.x,
                                                      y=object_loc.y - object.transform.location.y,
                                                      z=object_loc.z - object.transform.location.z))
        object_world_matrix = get_matrix(object.transform)
    else:
        bb_transform = carla.Transform(object.bounding_box.location)
        object_world_matrix = get_matrix(object.get_transform())
    bb_object_matrix = get_matrix(bb_transform)
    bb_world_matrix = np.dot(object_world_matrix, bb_object_matrix)
    world_cords = np.dot(bb_world_matrix, np.transpose(cords))
    return world_cords

def world_to_sensor(cords, sensor):
    """
    Transforms world coordinates to sensor.
    """
    sensor_world_matrix = get_matrix(sensor.get_transform())
    world_sensor_matrix = np.linalg.inv(sensor_world_matrix)
    sensor_cords = np.dot(world_sensor_matrix, cords)
    return sensor_cords

def get_matrix(transform):
    """
    Creates matrix from carla transform.
    """

    rotation = transform.rotation
    location = transform.location
    c_y = np.cos(np.radians(rotation.yaw))
    s_y = np.sin(np.radians(rotation.yaw))
    c_r = np.cos(np.radians(rotation.roll))
    s_r = np.sin(np.radians(rotation.roll))
    c_p = np.cos(np.radians(rotation.pitch))
    s_p = np.sin(np.radians(rotation.pitch))
    matrix = np.matrix(np.identity(4))
    matrix[0, 3] = location.x
    matrix[1, 3] = location.y
    matrix[2, 3] = location.z
    matrix[0, 0] = c_p * c_y
    matrix[0, 1] = c_y * s_p * s_r - s_y * c_r
    matrix[0, 2] = -c_y * s_p * c_r - s_y * s_r
    matrix[1, 0] = s_y * c_p
    matrix[1, 1] = s_y * s_p * s_r + c_y * c_r
    matrix[1, 2] = -s_y * s_p * c_r + c_y * s_r
    matrix[2, 0] = s_p
    matrix[2, 1] = -c_p * s_r
    matrix[2, 2] = c_p * c_r
    return matrix