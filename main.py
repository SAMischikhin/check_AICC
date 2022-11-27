import os
import time
import configparser
import warnings
from numpy import ComplexWarning
from solve import Solve
from messages import MyParsingError
from utils import get_input_file_dict, get_volumes_from_BXDATA

"""требуемый входной набор (имена файлов)"""
REQUIRED_INPUT_FILE_NAMES = ('H2_m', 'N2_m', 'O2_m', 'vapour_m', 'p', 't', 'BXDATA') #'CO_m', 

"""отключение FutureWarnings: версия конвертора в excel устарела;
отключение ComplexWarnings: решение полинома (функция get_P_AICC) откидываются"""
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ComplexWarning)

""" Создание объекта решения задачи"""
solve = Solve()

"""Чтение конфиг-файла"""
config = configparser.ConfigParser(
    converters={'list': lambda x: [float(i.strip()) for i in x.split(',')]})
try:
    config.read("config.ini", encoding="utf-8-sig")
    solve.myInfo.send('config.ini is readed')
except configparser.Error:
    raise MyParsingError("config.ini", logger)

""" задание пути к исходным данным, как атрибута класса;
копирование файлов в рабочую директорию осуществляться не будет"""
solve.get_input_dir_path(config)

"""проверка наличия директории с исходными данными (input_dir)"""
try:
    input_dir_list = os.listdir(solve.input_dir_path)
except:
    solve.myError.send("INPUT DIRECTORY DO NOT FOUND")

"""получение перечня файлов из input_dir, соответствующих требуемым именам"""
input_dir_dict_comp = get_input_file_dict(input_dir_list, REQUIRED_INPUT_FILE_NAMES)

"""проверка наличия всех требуемых файлов в директории входных данных"""
if  input_dir_dict_comp == None:
    solve.myError.send("Files with the following names(capital sensitive): {} should be in input directory".format(', '.join(REQUIRED_INPUT_FILE_NAMES)))
 
""" Чтение файлов """
solve.myInfo.send('Reading required input files from {}'.format(solve.input_dir_path))

""" Путь bxdata удобнее иметь в общем доступе """
bxdata_path = '{}\\{}'.format(solve.input_dir_path, input_dir_dict_comp.pop('BXDATA'))

""" Определение объемов боксов"""
volumes = get_volumes_from_BXDATA(bxdata_path)

""" Определение среднеобъемной температуры """
solve.set_tvol(input_dir_dict_comp.pop('t'), volumes)

""" Определение давления в реперном расчетном объеме """
solve.set_p(input_dir_dict_comp.pop('p'), config)

""" Копирование данных по концентрациям компонентов"""
solve.set_concentrations(input_dir_dict_comp, input_dir_list)

""" Проверка того, что pd.DataFrame.series имеют одинаковую длину """
solve.check_pd_series_length(solve.components_df)

""" Пересчет масс компонентов на каждом шаге в количество моль"""
solve.myInfo.send('Moles number is calculating now based on components masses')
solve.get_mol_fraction(config)

""" Присваиваем мультииндексы """
solve.set_multiindex()

""" Определение изменения состава смеси в предположении AICC (при сгорании CO и H2) """
solve.myInfo.send('Mixture composition changing due to AICC is calculating now')
solve.get_new_concentrations()

"""определение очередности колонок при записи в файл 'components'"""
solve.components_df = solve.components_df[["Time"] + solve.list_components]

"""Запись файла компонентов"""
solve.get_mass_fraction(solve.components_df, config).to_excel('{}\\components_out.xls'.format(solve.solve_path))
solve.myInfo.send('{} is saved'.format('components_out.xls'))

""" Определение теплоты сгорания CO и H2"""
solve.get_heat_of_combustion(config)

""" Определяем давление с учетом AICC"""
solve.myInfo.send('Pressure with considering AICC is calculating now')
solve.get_P_AICC(config)

"""Запись файла параметров"""
solve.parameters_df.to_excel('{}\\parameters_out.xls'.format(solve.solve_path))
solve.myInfo.send('{} is saved'.format('parameters_out.xls'))

solve.myInfo.send('Program is finished successfully')

time.sleep(5)
exit()