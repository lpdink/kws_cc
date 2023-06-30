#include <iostream>
using namespace std;

template <typename T>
class Myclass {
 public:
  Myclass(T data) : data_(data) {}
  void print();  //{cout<<this->data_<<endl;}
 private:
  T data_;
};

template <typename T>
void Myclass<T>::print() {
  cout << this->data_ << endl;
}

int main() {
  auto *ptr = new Myclass<float>(1.23);
  ptr->print();
  auto *ptr2 = new Myclass<int>(114514);
  ptr2->print();
}