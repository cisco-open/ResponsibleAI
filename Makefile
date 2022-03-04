all: build
.PHONY: build install

build:
	@python3 setup.py bdist_wheel

install: build
	@pip3 install dist/rai*.whl

uninstall:
	@pip3 uninstall -y rai

clean:
	@rm -rf build dist rai.egg-info
