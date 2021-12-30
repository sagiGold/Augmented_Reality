import numpy as np
import cv2
import trimesh
import pyrender


class MeshRenderer:
    def __init__(self, K, video_width, video_height, obj_path):
        self.K = K
        self.video_width = video_width
        self.video_height = video_height

        mesh = trimesh.load(obj_path)
        # normalize bounding box from (0,0,0) to max(30)
        mesh.rezero()  # set th LOWER LEFT (?) as (0,0,0)
        mesh
        T = np.eye(4)
        T[0:3, 0:3] = 10 * np.eye(3) * (1 / np.max(mesh.bounds))
        mesh.apply_transform(T)

        # rotate to make the drill standup
        T = np.eye(4)
        T[0:3, 0:3] = self.rot_x(np.pi / 2)
        mesh.apply_transform(T)

       # translation matrix to position pickle rick
        T = np.eye(4)
        T[0:3,3] = np.array([-6,9,0]) #nice
        mesh.apply_transform(T)

        # rotate 180 around x because the Z dir of the reference grid is down
        a =1.5
        T = np.eye(4)
        T[0:3, 0:3] = np.array([[a,0,0],[0,a,0],[0,0,a]])
        mesh.apply_transform(T)

        T = np.eye(4)
        T[0:3, 0:3] = self.rot_y(np.pi)
        mesh.apply_transform(T)

        T = np.eye(4)
        T[0:3, 0:3] = self.rot_x(np.pi)
        mesh.apply_transform(T)

        # Load the trimesh and put it in a scene
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene = pyrender.Scene(bg_color=np.array([0, 0, 0, 0]))
        scene.add(mesh)

        # add temp cam
        self.camera = pyrender.IntrinsicsCamera(
            self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2], zfar=10000, name="cam")
        light_pose = np.array(
            [
                [5.0, 0, 0, 10.0],
                [0, 5.0, 0.0, 100.0],
                [0.0, 0, 5.0, 10.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.cam_node = scene.add(self.camera, pose=light_pose)

        # Set up the light -- a single spot light in z+
        light = pyrender.PointLight(color=100 * np.ones(3), intensity=330.0)#, innerConeAngle=np.pi / 8.0)
        scene.add(light, pose=light_pose)

        self.scene = scene
        self.r = pyrender.OffscreenRenderer(self.video_width, self.video_height)
        # add the A flag for the masking
        self.flag = pyrender.constants.RenderFlags.RGBA

    def draw(self, img, rvec, tvec):
        # ===== update cam pose
        camera_pose = np.eye(4)
        res_R, _ = cv2.Rodrigues(rvec)

        # opengl transformation
        # https://stackoverflow.com/a/18643735/4879610
        camera_pose[0:3, 0:3] = res_R.T
        camera_pose[0:3, 3] = (-res_R.T @ tvec).flatten()
        # 180 about x
        camera_pose = camera_pose @ np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        self.scene.set_pose(self.cam_node, camera_pose)

        # ====== Render the scene
        color, depth = self.r.render(self.scene, flags=self.flag)
        img[color[:, :, 3] != 0] = color[:, :, 0:3][color[:, :, 3] != 0]
        return img

    def rot_x(self, t):
        ct = np.cos(t)
        st = np.sin(t)
        m = np.array([[1, 0, 0], [0, ct, -st], [0, st, ct]])
        return m

    def rot_y(self, t):
        ct = np.cos(t)
        st = np.sin(t)
        m = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])
        return m

    def rot_z(self, t):
        ct = np.cos(t)
        st = np.sin(t)
        m = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
        return m