from jetbotSim import Robot, Camera
import numpy as np
import cv2
import math
import config
import torch.nn.functional as F
import torchvision
import torch

class PIDController():
    def __init__(self, kp=1, ki=0.00001, kd=0.5, dt = 0.1):
        self.acc_ep = 0
        self.last_ep = 0
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.wheel_dist = config.wheel_dist
        self.wheel_rad = config.wheel_rad
        self.dt = dt
        self.v = 8
        self.w = 0
    
    def computeError(self, mask, image):
        img = image.copy()
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        error_ang = 0
        if len(contours) > 0:
            c = max(contours, key=cv2.contourArea)
            cv2.drawContours(img, contours, -1, (0,255,0), 3)

            min_x = c[c[:,:,0].argmin()][0][0]
            max_x = c[c[:,:,0].argmax()][0][0]
            y = int(mask.shape[0]*(5/6))
            x = int((min_x + max_x) / 2)

            line_center = (x, y)
            image_center = (int(mask.shape[1]/2), y)
            origin = (int(mask.shape[1]/2), 0)

            cv2.circle(img, line_center, 5, (255,255,0),-1)
            cv2.line(img, (image_center[0], mask.shape[1]), (image_center[0], 0), (255,255,255),3)

            delta_y = y
            delta_x = x - origin[0]
            ang = math.atan2(delta_y, delta_x)
            error_ang = ang - math.pi/2
        else:
            self.v = 0

        text = str("Error_ang: ") + str(round(error_ang*180/math.pi,3))
        cv2.putText(img, text, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)
        #cv2.imshow("error", img)
       
        return error_ang*180/math.pi, img
        
    def feedback(self, mask, img, avoid_ep):
        ep,error_img = self.computeError(mask, img)
        self.acc_ep += ep
        diff_ep = (ep - self.last_ep) / self.dt
        self.w = (self.kp*(ep+avoid_ep) + self.ki*self.acc_ep + self.kd*diff_ep)
        self.last_ep = ep 
        
        return error_img

    def update(self):
        right_v = self.v + self.w/(self.wheel_dist/2)
        left_v = self.v - self.w/(self.wheel_dist/2)

        robot.set_motor(left_v/(self.wheel_rad*2), right_v/(self.wheel_rad*2))

def preprocess(camera_value):
    global device, normalize
    x = camera_value
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x.transpose((2, 0, 1))
    x = torch.from_numpy(x).float()
    x = normalize(x)
    x = x.to(device)
    x = x[None, ...]
    return x

def centerRedLineDetection(img):
    global frames, target_brightness

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_blur = cv2.GaussianBlur(hsv, (3,3), 0)
    v = hsv_blur[:,:,2]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v = clahe.apply(v)
    hsv_blur[:,:,2] = v

    #brightness = np.sum(hsv_blur[:int(2*hsv_blur.shape[0]/3),:,2]) / 255 * int(2*hsv_blur.shape[0]/3) * hsv_blur.shape[1]
    #if frames < 10:
    #    target_brightness = brightness

    #hsv_blur = cv2.convertScaleAbs(hsv_blur, alpha = 1, beta = 255 * (target_brightness/brightness))

    #cv2.imshow("hsv_blur",hsv_blur)

    # The range of red region has to be further fine-tuned
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(hsv_blur, lower_red, upper_red)

    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(hsv_blur, lower_red, upper_red)  
    
    mask = mask0 + mask1

    kernel = np.ones((3,3), np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 1)
    mask = cv2.dilate(mask, kernel, iterations = 1)

    kernel = np.ones((7,7), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations = 1)
    mask = cv2.erode(mask, kernel, iterations = 1)
    mask[:int(2*mask.shape[0]/3),:] = 0

    res = cv2.bitwise_and(img, img, mask=mask)
    return res, mask

def execute(change):
    global robot, frames, pid, blocked_left, blocked_right
    img = cv2.resize(change["new"],(640,320))
    x = preprocess(img)
    y = model(x)
    y = F.softmax(y, dim=1)

    i = torch.argmax(y.flatten())
    prob = y.flatten()[i]

    if i == 0:
        text = "Left blocked ! " + str(round(prob.item(), 2)) 
    if i == 1:
        text = "Right blocked ! " + str(round(prob.item(), 2)) 
    if i == 2:
        text = "Free ! " + str(round(prob.item(), 2)) 
    cv2.putText(img, text, (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1, cv2.LINE_AA)

    #frames += 1
    #print(frames)

    #if frames % 20 == 0:
    #    cv2.imwrite("dataset/{}.jpg".format(frames), img)

    
    redLine, mask = centerRedLineDetection(img)
    if i != 2 and prob > 0.75:
        if i == 0:
            #print("block left detected")
            error_img = pid.feedback(mask, redLine, avoid_ep = -100)
            pid.update()
        elif i == 1:
            #print("block right detected")
            error_img = pid.feedback(mask, redLine, avoid_ep = 100)
            pid.update()
    else:
        error_img = pid.feedback(mask, redLine, avoid_ep = 0)
        pid.update()

    frames += 1 
    res = cv2.hconcat([img, error_img])
    cv2.imshow("camera", res)

frames = 0
target_brightness = 0

#model = torchvision.models.mobilenet_v2(pretrained=False)
model = torchvision.models.alexnet(pretrained=False)
model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, 3)
#model.load_state_dict(torch.load('best_avoidance_model_MobileNetv2.pth'))
model.load_state_dict(torch.load('best_avoidance_model_alexnet.pth'))

device = torch.device('cuda')
model = model.to(device)

mean = 255.0 * np.array([0.485, 0.456, 0.406])
stdev = 255.0 * np.array([0.229, 0.224, 0.225])

normalize = torchvision.transforms.Normalize(mean, stdev)

robot = Robot()
camera = Camera()
pid = PIDController()
camera.observe(execute)