def dec1(func):
    print("HHHA:0====>")

    def one():
        print("HHHA:0.1====>")
        func()
        print("HHHA:0.2====>")

    return one


def dec2(func):
    print("HHHB:0====>")

    def two():
        print("HHHB:0.1====>")
        func()
        print("HHHB:0.2====>")

    return two


def dec3(func):
    print("HHHC:0====>")

    def three():
        print("HHHC:0.1====>")
        func()
        print("HHHC:0.2====>")

    return three


@dec1
@dec2
@dec3
def test():
    print("HHHD:0====>test")


print("HHHH:0====>")
# test()