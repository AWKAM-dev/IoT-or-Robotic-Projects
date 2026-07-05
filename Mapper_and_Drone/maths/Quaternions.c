#include <stdio.h>
#include <math.h>

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

void EulerToQuaternion(struct EulerAngles euler, struct Quaternion *quat) {
    //Convert degrees to radians
    float roll = euler.roll * (3.14159265358979323846f / 180.0f);
    float pitch = euler.pitch * (3.14159265358979323846f / 180.0f);
    float yaw = euler.yaw * (3.14159265358979323846f / 180.0f);

    // Calculate the quaternion components
    float cy = cos(yaw * 0.5f);
    float sy = sin(yaw * 0.5f);
    float cp = cos(pitch * 0.5f);
    float sp = sin(pitch * 0.5f);
    float cr = cos(roll * 0.5f);
    float sr = sin(roll * 0.5f);

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