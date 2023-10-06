all: build
.PHONY: build install

build:
	@python3 setup.py bdist_wheel

install: build
	@pip3 install dist/OpenRAI*.whl

uninstall:
	@pip3 uninstall -y OpenRAI

clean:
	@rm -rf build dist OpenRAI.egg-info
