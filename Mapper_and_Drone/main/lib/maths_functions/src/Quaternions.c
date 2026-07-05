#include <stdio.h>
#include <math.h>

#define M_PI_F 3.14159265358979323846f

struct EulerAngles {
    float roll;  // Rotation around the x-axis. In degrees
    float pitch; // Rotation around the y-axis. In degrees
    float yaw;   // Rotation around the z-axis. In degrees
}EulerAngles;

struct Quaternion {
    float w; // Scalar part
    float x; // x component of the vector part
    float y; // y component of the vector part
    float z; // z component of the vector part
}Quaternion;

struct Quaternion euler_to_quaternion(struct EulerAngles euler, struct Quaternion *quat) {
    // Convert degrees to radians
    float roll  = euler.roll  * (M_PI_F / 180.0f);
    float pitch = euler.pitch * (M_PI_F / 180.0f);
    float yaw   = euler.yaw   * (M_PI_F / 180.0f);
    
    float cy = cosf(yaw * 0.5f);
    float sy = sinf(yaw * 0.5f);
    float cp = cosf(pitch * 0.5f);
    float sp = sinf(pitch * 0.5f);
    float cr = cosf(roll * 0.5f);
    float sr = sinf(roll * 0.5f);

    quat->w = cr * cp * cy + sr * sp * sy;
    quat->x = sr * cp * cy - cr * sp * sy;
    quat->y = cr * sp * cy + sr * cp * sy;
    quat->z = cr * cp * sy - sr * sp * cy;
}

int main(){
    printf("Eulers to Quaternions\n");

    struct EulerAngles angle;
    angle.roll = 0.0f;
    angle.pitch = 0.0f;
    angle.yaw = 0.0f;

    struct Quaternion quat;
    EulerToQuaternion(angle, &quat);
    printf("Quaternion: w = %f, x = %f, y = %f, z = %f\n", quat.w, quat.x, quat.y, quat.z);

    return 0;
}