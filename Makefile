all: build
.PHONY: build install

build:
	@python3 setup.py bdist_wheel

install: build
	@pip3 install dist/py-rai*.whl

uninstall:
	@pip3 uninstall -y py-rai

clean:
	@rm -rf build dist py-rai.egg-info
