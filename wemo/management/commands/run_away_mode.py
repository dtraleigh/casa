import random
from datetime import datetime, timedelta, time
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.db import transaction
from wemo.models import WemoSwitch, AwayModeSettings
from astral import LocationInfo
from astral.sun import sun

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
        eastern = timezone.pytz.timezone('America/New_York')
        now = timezone.now().astimezone(eastern)
        today = now.date()

        # Get sunset time for today
        s = sun(city.observer, date=today, tzinfo=eastern)
        sunset_time = s['sunset']

        logger.info(f"Today's sunset: {sunset_time.strftime('%I:%M %p %Z')}")
        logger.info(f"Current time: {now.strftime('%I:%M %p %Z')}")

        # Check if we should turn lights ON (around sunset)
        if settings.last_sunset_on != today:
            self.check_sunset_on(settings, sunset_time, now, options['dry_run'])
        else:
            logger.info("Already turned lights on at sunset today")

        # Check if we should turn lights OFF (around 10:30 PM)
        if settings.last_night_off != today:
            self.check_night_off(settings, now, eastern, options['dry_run'])
        else:
            logger.info("Already turned lights off tonight")

    def check_sunset_on(self, settings, sunset_time, now, dry_run):
        """Check if it's time to turn lights on around sunset."""
        # Calculate the random window
        window_start = sunset_time - timedelta(minutes=settings.sunset_window_minutes)
        window_end = sunset_time + timedelta(minutes=settings.sunset_window_minutes)

        logger.info(
            f"Sunset ON window: {window_start.strftime('%I:%M %p')} - {window_end.strftime('%I:%M %p')}")

        if window_start <= now <= window_end:
            # We're in the window! Turn on lights at a random time
            # Calculate if we should turn on now or wait
            total_window_seconds = (window_end - window_start).total_seconds()
            elapsed_seconds = (now - window_start).total_seconds()

            # Use a probability based on how far we are in the window
            # This ensures lights turn on sometime during the window
            probability = elapsed_seconds / total_window_seconds

            if random.random() < probability or elapsed_seconds > (total_window_seconds * 0.8):
                logger.info(self.style.WARNING("TIME TO TURN LIGHTS ON!"))

                if dry_run:
                    logger.info(self.style.WARNING("DRY RUN - Would turn on lights now"))
                else:
                    self.turn_on_all_switches()
                    settings.last_sunset_on = now.date()
                    settings.save()
                    logger.info(self.style.SUCCESS("Lights turned ON"))
            else:
                logger.info(f"In sunset window but waiting (probability: {probability:.2%})")
        elif now < window_start:
            minutes_until = int((window_start - now).total_seconds() / 60)
            logger.info(f"Sunset window starts in {minutes_until} minutes")
        else:
            logger.info(self.style.ERROR("Missed sunset window for today"))

    def check_night_off(self, settings, now, eastern, dry_run):
        """Check if it's time to turn lights off around 10:30 PM."""
        # Create the target off time for today
        off_time = eastern.localize(datetime.combine(
            now.date(),
            time(settings.off_time_hour, settings.off_time_minute)
        ))

        window_start = off_time - timedelta(minutes=settings.off_window_minutes)
        window_end = off_time + timedelta(minutes=settings.off_window_minutes)

        logger.info(
            f"Night OFF window: {window_start.strftime('%I:%M %p')} - {window_end.strftime('%I:%M %p')}")

        if window_start <= now <= window_end:
            # We're in the window! Turn off lights at a random time
            total_window_seconds = (window_end - window_start).total_seconds()
            elapsed_seconds = (now - window_start).total_seconds()

            probability = elapsed_seconds / total_window_seconds

            if random.random() < probability or elapsed_seconds > (total_window_seconds * 0.8):
                logger.info(self.style.WARNING("TIME TO TURN LIGHTS OFF!"))

                if dry_run:
                    logger.info(self.style.WARNING("DRY RUN - Would turn off lights now"))
                else:
                    self.turn_off_all_switches()
                    settings.last_night_off = now.date()
                    settings.save()
                    logger.info(self.style.SUCCESS("Lights turned OFF"))
            else:
                logger.info(f"In night window but waiting (probability: {probability:.2%})")
        elif now < window_start:
            minutes_until = int((window_start - now).total_seconds() / 60)
            logger.info(f"Night window starts in {minutes_until} minutes")
        else:
            logger.info("Past night window for today")

    def turn_on_all_switches(self):
        """Turn on all enabled Wemo switches."""
        switches = WemoSwitch.objects.filter(disabled=False)

        for switch in switches:
            try:
                switch.turn_on()
                logger.info(f"  ✓ Turned ON: {switch.name}")
            except Exception as e:
                logger.info(self.style.ERROR(f"Failed to turn on {switch.name}: {e}"))

    def turn_off_all_switches(self):
        """Turn off all enabled Wemo switches."""
        switches = WemoSwitch.objects.filter(disabled=False)

        for switch in switches:
            try:
                switch.turn_off()
                logger.info(f"Turned OFF: {switch.name}")
            except Exception as e:
                logger.info(self.style.ERROR(f"Failed to turn off {switch.name}: {e}"))