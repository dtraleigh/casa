import random
from datetime import datetime, timedelta, time
from django.core.management.base import BaseCommand
from django.utils import timezone
from wemo.models import WemoSwitch, AwayModeSettings, SwitchAwaySchedule
from astral import LocationInfo
from astral.sun import sun
from zoneinfo import ZoneInfo

import logging
logger = logging.getLogger('away_mode')


class Command(BaseCommand):
    help = 'Run Away Mode logic to control lights based on sunset and scheduled times'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Show what would happen without making changes',
        )

    def handle(self, *args, **options):
        settings = AwayModeSettings.get_settings()

        if not settings.enabled:
            logger.info("Away Mode is disabled")
            return

        logger.info("Away Mode is ENABLED")

        # Set up location for North Carolina (using Raleigh as reference)
        city = LocationInfo("Raleigh", "USA", "America/New_York", 35.7796, -78.6382)

        # Get today's date in Eastern timezone
        eastern = ZoneInfo("America/New_York")
        now = timezone.now().astimezone(eastern)
        today = now.date()

        # Get sunset time for today
        s = sun(city.observer, date=today, tzinfo=eastern)
        sunset_time = s['sunset']

        logger.info(f"Today's sunset: {sunset_time.strftime('%I:%M %p %Z')}")
        logger.info(f"Current time: {now.strftime('%I:%M %p %Z')}")

        dry_run = options['dry_run']

        # Ensure schedules exist for today
        switches = WemoSwitch.objects.filter(disabled=False)
        self.ensure_schedules(settings, switches, today, sunset_time, eastern)

        # Execute any scheduled on actions that are due
        self.execute_scheduled_on(today, now, dry_run)

        # Execute any scheduled off actions that are due
        self.execute_scheduled_off(today, now, dry_run)

    def ensure_schedules(self, settings, switches, today, sunset_time, eastern):
        """Create schedules for today if they don't exist yet."""
        for switch in switches:
            schedule, created = SwitchAwaySchedule.objects.get_or_create(
                switch=switch,
                date=today,
            )
            if created:
                # Generate random on time within sunset window
                sunset_window_start = sunset_time - timedelta(minutes=settings.sunset_window_minutes)
                sunset_window_end = sunset_time + timedelta(minutes=settings.sunset_window_minutes)
                total_on_seconds = int((sunset_window_end - sunset_window_start).total_seconds())
                random_on_offset = random.randint(0, total_on_seconds)
                schedule.planned_on_time = sunset_window_start + timedelta(seconds=random_on_offset)

                # Generate random off time within off-time window
                off_time = datetime.combine(
                    today,
                    time(settings.off_time_hour, settings.off_time_minute),
                    tzinfo=eastern
                )
                off_window_start = off_time - timedelta(minutes=settings.off_window_minutes)
                off_window_end = off_time + timedelta(minutes=settings.off_window_minutes)
                total_off_seconds = int((off_window_end - off_window_start).total_seconds())
                random_off_offset = random.randint(0, total_off_seconds)
                schedule.planned_off_time = off_window_start + timedelta(seconds=random_off_offset)

                schedule.save()
                logger.info(
                    f"  Scheduled {switch.name}: "
                    f"ON at {schedule.planned_on_time.strftime('%I:%M %p')}, "
                    f"OFF at {schedule.planned_off_time.strftime('%I:%M %p')}"
                )

    def execute_scheduled_on(self, today, now, dry_run):
        """Turn on switches whose planned_on_time has passed."""
        due_schedules = SwitchAwaySchedule.objects.filter(
            date=today,
            on_executed=False,
            planned_on_time__lte=now,
        ).select_related('switch')

        for schedule in due_schedules:
            switch = schedule.switch
            if switch.disabled:
                continue

            logger.info(self.style.WARNING(
                f"TIME TO TURN ON: {switch.name} "
                f"(scheduled {schedule.planned_on_time.strftime('%I:%M %p')})"
            ))

            if dry_run:
                logger.info(self.style.WARNING(f"  DRY RUN - Would turn on {switch.name}"))
            else:
                try:
                    switch.turn_on(notes="Away Mode")
                    schedule.on_executed = True
                    schedule.save(update_fields=['on_executed'])
                    logger.info(self.style.SUCCESS(f"  Turned ON: {switch.name}"))
                except Exception as e:
                    logger.info(self.style.ERROR(f"  Failed to turn on {switch.name}: {e}"))

    def execute_scheduled_off(self, today, now, dry_run):
        """Turn off switches whose planned_off_time has passed."""
        due_schedules = SwitchAwaySchedule.objects.filter(
            date=today,
            off_executed=False,
            planned_off_time__lte=now,
        ).select_related('switch')

        for schedule in due_schedules:
            switch = schedule.switch
            if switch.disabled:
                continue

            logger.info(self.style.WARNING(
                f"TIME TO TURN OFF: {switch.name} "
                f"(scheduled {schedule.planned_off_time.strftime('%I:%M %p')})"
            ))

            if dry_run:
                logger.info(self.style.WARNING(f"  DRY RUN - Would turn off {switch.name}"))
            else:
                try:
                    switch.turn_off(notes="Away Mode")
                    schedule.off_executed = True
                    schedule.save(update_fields=['off_executed'])
                    logger.info(self.style.SUCCESS(f"  Turned OFF: {switch.name}"))
                except Exception as e:
                    logger.info(self.style.ERROR(f"  Failed to turn off {switch.name}: {e}"))