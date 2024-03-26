#include <iostream>

int main() {
    // 假设我们有一个8比特的数
    unsigned char eight_bit_number = 0xAB; // 二进制为 10101011

    // 提取两个4比特的数
    unsigned char first_four_bits = eight_bit_number >> 4; // 右移4位来获得高4位
    unsigned char second_four_bits = eight_bit_number & 0x0F; // 使用位与运算和掩码来获得低4位

    // 输出结果
    std::cout << "8比特的数: " << std::hex << static_cast<int>(eight_bit_number) << std::endl;
    std::cout << "第一个4比特的数: " << std::hex << static_cast<int>(first_four_bits) << std::endl;
    std::cout << "第二个4比特的数: " << std::hex << static_cast<int>(second_four_bits) << std::endl;

    return 0;
}
