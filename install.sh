#!/bin/bash

check_virtualenv() {
    if ! command -v virtualenv &> /dev/null; then
        echo "virtualenv is not installed. Installing..."
        sudo python3.9 -m pip install --user virtualenv
        echo "virtualenv installation complete."
    fi
}

export_deps() {
    local env_name=${1:-".ssdwsn_venv"}

    if [ ! -d "$env_name" ]; then
        echo "Virtual environment '$env_name' not found. Use '$0 create [env_name]' to create one."
        return 1
    fi

    source "./$env_name/bin/activate"
    pipreqs --force
    # pip freeze > requirements.txt
    echo "Dependencies exported to requirements.txt"
}

install() {
    # Check if virtualenv is installed, if not, install it
    pip install -U pip
    pip install pipreqs
    pip install Cython
    check_virtualenv
    
    local env_name=${1:-".ssdwsn_venv"}
    if [ -d "$env_name" ]; then
        echo "Virtual environment '$env_name' already exists. Aborting."
        source "./$env_name/bin/activate"

        if [ -f "requirements.txt" ]; then
            pip install -r ./requirements.txt
        fi
        if [ -f "setup.py" ]; then
            pip install -e .
        fi
        return 1
    fi

    sudo python3.9 -m venv "$env_name"
    sudo chmod -R 777 "."
    source "./$env_name/bin/activate"

    if [ -f "requirements.txt" ]; then
        pip install -r ./requirements.txt
    fi
    if [ -f "setup.py" ]; then
        pip install -e .
    fi    
}

run() {
    local env_name=${1:-".ssdwsn_venv"}

    if [ ! -d "$env_name" ]; then
        echo "Virtual environment '$env_name' not found. Use '$0 install [env_name]' to install SSDWSN."
        return 1
    fi
    lsof -nti:6006 | xargs kill -9
    lsof -nti:4455 | xargs kill -9
    source "./$env_name/bin/activate"
    echo which python
    sudo "./$env_name/lib/python3.9" setup.py install
    sudo chmod -R 777 "."
    tensorboard --logdir output/logs &
    "./$env_name/lib/python3.9" ssdwsn/util/plot/app.py &
    sudo "./$env_name/lib/python3.9" ssdwsn/main.py
}

clean() {

    local env_name=${1:-".ssdwsn_venv"}
    sudo rm -r build
    sudo rm -r dist
    sudo rm -r ssdwsn.egg-info
    sudo rm -r outputs/logs/*
    sudo rm -r ssdwsn/__pycache__
    sudo rm -r ssdwsn/app/__pycache__
    sudo rm -r ssdwsn/ctrl/__pycache__
    sudo rm -r ssdwsn/data/__pycache__
    sudo rm -r ssdwsn/openflow/__pycache__
    sudo rm -r ssdwsn/util/__pycache__
    lsof -nti:6006 | xargs kill -9
    lsof -nti:4455 | xargs kill -9
    "pkill -9 ./$env_name/lib/python3.9 | kill -9 $(ps -A | grep python | awk '{print $1}')"
}

uninstall() {
    local env_name=${1:-".ssdwsn_venv"}

    if [ ! -d "$env_name" ]; then
        echo "Virtual environment '$env_name' not found."
        return 1
    fi
    
    clean
    deactivate
    sudo rm -rf "$env_name"
}

print_help() {
    echo "Usage: $0 [option] [env_name]"
    echo "Options:"
    echo "  install      Install SSDWSN"
    echo "  run      Run SSDWSN"
    echo "  clean      clean SSDWSN setup files and logs"
    echo "  uninstall    Uninstall SSDWSN"
    echo "  export      Export installed dependencies to requirements.txt within a virtual environment (default name: .ssdwsn_venv)"
}

if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
    print_help
    return 0
fi

case "$1" in
    "install")
        install "$2"
        ;;
    "run")
        run "$2"
        ;;
    "clean")
        clean "$2"
        ;;
    "uninstall")
        uninstall "$2"
        ;;
    "export")
        export_deps "$2"
        ;;
    *)
        echo "Unknown option: $1"
        print_help
        exit 1
        ;;
esac